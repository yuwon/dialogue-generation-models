import grpc

import chat_pb2
import chat_pb2_grpc


def run():
    with grpc.insecure_channel("ai.vricarus.com:50081") as channel:
        stub = chat_pb2_grpc.ChattingStub(channel)
        my_message = chat_pb2.Message(text="안녕 뭐하니")
        response = stub.Chat(request=my_message)
        print("response:", response.text)


if __name__ == "__main__":
    run()
