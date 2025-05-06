import json
from typing import Optional

import fire
import logging
import torch
import torch.multiprocessing as mp

from llama import Llama


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


PROMPT = """
Which EARL (Emotion Annotation and Representation Language) emotion best describes the following?

"{}"

please answer with a single word
"""


SYSTEM_PROMPT = """
EARL (Emotion Annotation and Representation Language) emotions are listed here in nested json format:

    "joy": ["happiness", "amusement", "pleasure", "pride", "relief", "satisfaction"],
    "anger": ["annoyance", "irritation", "rage", "disgust", "envy", "torment"],
    "sadness": ["disappointment", "shame", "neglect", "suffering", "guilt", "depression"],
    "fear": ["nervousness", "horror", "worry", "insecurity", "panic"],
    "surprise": ["shock", "amazement", "wonder", "disbelief"],
    "disgust": ["revulsion", "aversion", "contempt"],
    "love": ["affection", "longing", "infatuation"],
    "hope": ["optimism", "anticipation", "encouragement"],
    "pride": ["triumph", "self-respect", "dignity"],
    "shame": ["embarrassment", "humiliation", "remorse"]

Please select only one of the EARL emotions (both keys and values are valid) as your response.
"""


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
):
    device = torch.device(f"cuda:{rank}")

    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
    )

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
            [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT.format(d["response"])},
                ]
                for d in dialogs
            ],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        for dialog, result in zip(dialogs, results):
            res = dialog
            res["guess"] = result["generation"]["content"]
            result_queue.put(res)
        del dialogs, results
        logger.info("Inferred a batch on GPU %i", rank)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    responses_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    out_file: str = "/dev/null",
    devices: int = 8,
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
            ),
        )
        p.start()
        processes.append(p)

    with open(responses_path, "r") as f:
        responses = json.load(f)

    i = 0
    while i < len(responses):
        dialogs = responses[i : min(len(responses), i + max_batch_size)]
        logger.info("Sent dialog to queue, now of size: %i", prompt_queue.qsize())
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
