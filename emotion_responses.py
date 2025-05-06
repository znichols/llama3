import json
from typing import List, Optional

import fire
import logging
import torch
import torch.multiprocessing as mp

from llama import Dialog, Llama


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


earl_emotions = {
    "joy": ["happiness", "amusement", "pleasure", "pride", "relief", "satisfaction"],
    "anger": ["annoyance", "irritation", "rage", "disgust", "envy", "torment"],
    "sadness": [
        "disappointment",
        "shame",
        "neglect",
        "suffering",
        "guilt",
        "depression",
    ],
    "fear": ["nervousness", "horror", "worry", "insecurity", "panic"],
    "surprise": ["shock", "amazement", "wonder", "disbelief"],
    "disgust": ["revulsion", "aversion", "contempt"],
    "love": ["affection", "longing", "infatuation"],
    "hope": ["optimism", "anticipation", "encouragement"],
    "pride": ["triumph", "self-respect", "dignity"],
    "shame": ["embarrassment", "humiliation", "remorse"],
}


prompts = [
    (
        "I want you to write dialog for a character. "
        "The character is seeing someone for the first time in a long time, and is expressing the emotion {}. "
        "Please write three sentences with no preamble"
    ),
    (
        "I want you to write dialog for a character. "
        "The character is seeing someone for the first time in a long time, and is expressing the emotion {}. "
        "Please write one short sentence with no preamble"
    ),
]


def infer(
    rank,
    prompt_queue,
    result_queue,
    ckpt_dir,
    tokenizer_path,
    max_gen_len,
    temperature,
    top_p,
    max_seq_len,
    max_batch_size,
    noise_std: Optional[float] = None,
):
    device = torch.device(f"cuda:{rank}")

    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
    )

    if noise_std is not None:
        cur_seed = torch.initial_seed()
        torch.manual_seed(0)
        layer_copy = model.model.layers[-1].feed_forward.w3.weight.clone()
        layer_shape = layer_copy.shape
        noise = torch.randn(layer_shape) * noise_std
        model.model.layers[-1].feed_forward.w3.weight = torch.nn.Parameter(
            layer_copy + noise
        )
        torch.manual_seed(cur_seed)

    while True:
        dialogs = prompt_queue.get()
        if dialogs is None:
            logger.info("Received STOP signal on GPU %i", rank)
            break
        logger.info(
            "Received a batch on GPU %i; input queue size: %i; output queue size: %i",
            rank,
            prompt_queue.qsize(),
            result_queue.qsize(),
        )
        results = model.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        for dialog, result in zip(dialogs, results):
            result_queue.put(
                {
                    "prompt": dialog[0]["content"],
                    "response": result["generation"]["content"],
                    "temp": temperature,
                    "top_p": top_p,
                }
            )
        del dialogs, results
        logger.info("Inferred a batch on GPU %i", rank)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    n_gens: int = 100,
    out_file: str = "/dev/null",
    devices: int = 8,
    noise_std: Optional[float] = None,
):
    manager = mp.Manager()
    prompt_queue = manager.Queue()
    result_queue = manager.Queue()

    processes = []
    for rank in range(devices):
        p = mp.Process(
            target=infer,
            args=(
                rank,
                prompt_queue,
                result_queue,
                ckpt_dir,
                tokenizer_path,
                max_gen_len,
                temperature,
                top_p,
                max_seq_len,
                max_batch_size,
                noise_std,
            ),
        )
        p.start()
        processes.append(p)

    all_dialogs: List[Dialog] = []
    for e in earl_emotions.keys():
        all_dialogs += [
            [{"role": "user", "content": prompt.format(e)}] for prompt in prompts
        ]
        all_dialogs += [
            [{"role": "user", "content": prompt.format(sub_e)}]
            for prompt in prompts
            for sub_e in earl_emotions[e]
        ]

    for _ in range(n_gens):
        i = 0
        while i < len(all_dialogs):
            dialogs = all_dialogs[i : min(len(all_dialogs), i + max_batch_size)]
            logger.info(f"Sent dialog to queue, now of size: {prompt_queue.qsize()}")
            i += max_batch_size
            prompt_queue.put(dialogs)

    for _ in range(devices):
        prompt_queue.put(None)
    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    with open(out_file, "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    fire.Fire(main)
