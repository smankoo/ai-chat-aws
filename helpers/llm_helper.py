# Use the Conversation API to send a text message to Anthropic Claude
# and print the response stream.

import boto3
from botocore.exceptions import ClientError
from pprint import pprint


class Conversation:
    def __init__(self, message=None):
        self.blocks = []
        if message is not None:
            self.add_block("user", message)

    def add_block(self, role, message):
        self.blocks.append({"role": role, "content": [{"text": message}]})

    def get_blocks(self):
        return self.blocks


def call_llm(
    conversation: Conversation, aws_access_key_id=None, aws_secret_access_key=None
):

    # Create a Bedrock Runtime client in the AWS Region you want to use.
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Set the model ID, e.g., Claude 3 Haiku.
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    conversation_blocks = conversation.get_blocks()

    # Send the message to the model, using a basic inference configuration.
    streaming_response = client.converse_stream(
        modelId=model_id,
        messages=conversation_blocks,
        inferenceConfig={"maxTokens": 512, "temperature": 0, "topP": 0.9},
    )

    # Extract and print the streamed response text in real-time.
    for chunk in streaming_response["stream"]:
        if "contentBlockDelta" in chunk:
            text = chunk["contentBlockDelta"]["delta"]["text"]
            # print(text, end="")

            yield (text)


if __name__ == "__main__":
    convo = Conversation("Hello, How are you?")
    response = call_llm(convo)

    assistant_message = ""
    # print response which is a generator object
    for r in response:
        assistant_message += r
        print(r, end="")

    convo.add_block("assistant", assistant_message)
    convo_blocks = convo.get_blocks()
    pprint(convo_blocks)
