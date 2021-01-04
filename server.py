from concurrent import futures
import argparse
import logging

import grpc
import numpy as np
import sentencepiece as spm
import torch

from dialogue_generation_models.configuration_meena import MeenaConfig
from dialogue_generation_models.modeling_meena import MeenaForConditionalGeneration
import chat_pb2
import chat_pb2_grpc



# Setup logger
fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
date_fmt = "%H:%M:%S"
formatter = logging.Formatter(fmt, datefmt=date_fmt)

handler = logging.FileHandler(filename="./output.log", encoding="utf8")
handler.setFormatter(formatter)

logger = logging.getLogger("dialogue-generation-models")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def postprocess(text):
    text = text.replace("ㅋ", "")\
        .replace("ㅎ", "하")\
        .replace("^^", "")\
        .replace("ㅠ", "")\
        .replace("ㅜ", "")\
        .replace("[UNK]", "")\
        .replace("ㅇㅇ", "응")\
        .replace("ㄴ", "노")\
        .replace("얔", "야")\
        .replace("닠", "니")\
        .replace("ㅡㅡ^", "")\
        .replace("ㅡㅡ", "")\
        .replace("  ", " ")\
        .replace("개바쁨", "바쁨")\
        .replace("개소름", "소름")\
        .replace("알겟", "알겠")\
        .replace("겟다", "겠다")\
        .replace("겟어", "겠어")\
        .replace("겠답", "겠다")\
        .replace("모야", "뭐야")\
        .replace("모얌", "뭐야")\
        .replace("뭐얌", "뭐야")\
        .replace("기여", "귀여")\
        .replace("거얌", "거야")\
        .replace(";", "")\
        .replace("아님?", "아니야?")\
        .replace("싶엉", "싶어")\
        .replace("ㅇ", "응")\
        .replace("엇던", "었던")\
        .replace("까욤", "까요")\
        .replace("보냇", "보냈")\
        .replace("아니얌", "아니야")\
        .replace("엇어", "었어")\
        .replace("햇거든", "했거든")\
        .strip()
    return text


class ChattingServicer(chat_pb2_grpc.ChattingServicer):
    def __init__(self, pretrained_model_path, model_config_path, tokenizer_model_path, decoding_method):
        self.decoding_method = decoding_method
        logger.info("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        self.config = MeenaConfig.from_json(model_config_path)
        self.model = MeenaForConditionalGeneration(self.config)
        self.model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        self.model.eval()
        self.model.to(self.device)
        logger.info("Loading model complete.")

    def Chat(self, request, context):
        context = [request.text]
        input_ids = (
            torch.tensor(
                [
                    token_id
                    for utterance in context
                    for token_id in self.tokenizer.encode(utterance, out_type=int) + [self.config.sept_token_id]
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

        if self.decoding_method == "top_p":
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=48,
                min_length=8,
                temperature=1.0,
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.config.pad_token_id,
                bos_token_id=self.config.bos_token_id,
                eos_token_id=self.config.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=10,
            )
        elif self.decoding_method == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                max_length=48,
                min_length=8,
                num_beams=10,
                pad_token_id=self.config.pad_token_id,
                bos_token_id=self.config.bos_token_id,
                eos_token_id=self.config.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=10,
            )
        else:
            raise ValueError(f"Invalid decoding method: {self.decoding_method}")

        output = self.tokenizer.decode(outputs.tolist()[0])
        output = postprocess(output)
        logger.info("%s <--> %s", request.text, output)
        response = chat_pb2.Message(text=output)
        return response


def serve(args):
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=args.max_workers))
    chat_pb2_grpc.add_ChattingServicer_to_server(
        servicer=ChattingServicer(
            args.pretrained_model_path,
            args.model_config_path,
            args.tokenizer_model_path,
            args.decoding_method,
        ),
        server=server
    )
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="0.0.0.0", type=str,
    )
    parser.add_argument(
        "--port", default=50081, type=int,
    )
    parser.add_argument(
        "--max_workers", default=10, type=int,
        help="the maximum number of threads",
    )
    parser.add_argument(
        "--pretrained_model_path", default="./large_meena_trained_on_filtered_data_kr.pth", type=str,
        help="path to pre-trained model",
    )
    parser.add_argument(
        "--model_config_path", default="./configs/large_meena_config.json", type=str,
        help="path to model configuration file",
    )
    parser.add_argument(
        "--tokenizer_model_path", default="./tokenizer/kr_spm.model", type=str,
        help="path to sentencepiece model",
    )
    parser.add_argument(
        "--decoding_method", default="top_p", type=str,
        help="decoding method (beam_search or top_p)",
    )
    serve(parser.parse_args())
