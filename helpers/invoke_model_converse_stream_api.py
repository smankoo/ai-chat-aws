import boto3
import os
import json
from dotenv import load_dotenv

# loading in variables from .env file
load_dotenv()

# instantiating the Bedrock client, and passing in the CLI profile
boto3.setup_default_session(profile_name=os.getenv("profile_name"))
bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')

# Instantiate a messages list to store the conversation history
messages = []

# Define the tool configuration
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ and WKRP."
                            }
                        },
                        "required": ["sign"]
                    }
                },
                "name": "calc",
                "description": "Perform a calculation.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "operator": {
                                "type": "string",
                                "description": "The operator to use for the calculation. Example operators are +, -, *, /, %, **."
                            },
                            "operand1": {
                                "type": "number",
                                "description": "The first operand for the calculation."
                            },
                            "operand2": {
                                "type": "number",
                                "description": "The second operand for the calculation."
                            }
                        },
                        "required": ["operator", "operand1", "operand2"]
                    }
                }
            }
        }
    ]
}

def get_top_song(call_sign):
    """Returns the most popular song for the requested station."""
    if call_sign == 'WZPZ':
        return "Elemental Hotel", "8 Storey Hike"
    else:
        raise ValueError(f"Station {call_sign} not found.")

# write a calc function that takes in an operator and two operands and returns the output
def calc(operator, operand1, operand2):
    """Returns result of mathematical operation"""
    if operator == '+':
        return operand1 + operand2
    elif operator == '-':
        return operand1 - operand2
    elif operator == '*':
        return operand1 * operand2
    elif operator == '/':
        return operand1 / operand2
    elif operator == '%':
        return operand1 % operand2
    elif operator == '**':
        return operand1 ** operand2
    else:
        raise ValueError(f"Operator {operator} not supported.")

def stream_conversation(user_message):
    """
    Sends messages to a model and streams back the response.
    Args:
        user_message: The latest user message as a string.
        
    Yields:
        Streamed response chunks from the model.
    """
    temperature = 0.5
    top_k = 200
    inference_config = {"temperature": temperature}
    additional_model_fields = {"top_k": top_k}
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    system_prompts = [{"text": "You are a helpful assistant."
                                "If you have to do a mathematical calculation, you must do it using the tools provided."}]
    
    user_message_dict = {
        "role": "user",
        "content": [{"text": user_message}]
    }
    
    messages.append(user_message_dict)

    response = bedrock.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
        toolConfig=tool_config
    )

    tool_use = {}
    text = ""
    content = []

    for chunk in response['stream']:
        print("Chunk received:", chunk)  # Debug log to trace events

        if 'messageStart' in chunk:
            message = {'role': chunk['messageStart']['role']}
        elif 'contentBlockStart' in chunk:
            if 'toolUse' in chunk['contentBlockStart']['start']:
                tool = chunk['contentBlockStart']['start']['toolUse']
                tool_use['toolUseId'] = tool['toolUseId']
                tool_use['name'] = tool['name']
        elif 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                if 'input' not in tool_use:
                    tool_use['input'] = ''
                tool_use['input'] += delta['toolUse']['input']
            elif 'text' in delta:
                text += delta['text']
                yield delta['text']
        elif 'contentBlockStop' in chunk:
            if 'input' in tool_use:
                tool_use['input'] = json.loads(tool_use['input'])
                content.append({'toolUse': tool_use})
                tool_use = {}
            else:
                content.append({'text': text})
                text = ''
        elif 'messageStop' in chunk:
            stop_reason = chunk['messageStop']['stopReason']
            assistant_message = {'role': 'assistant', 'content': content}
            messages.append(assistant_message)

            if stop_reason == "tool_use":
                for content_block in content:
                    if 'toolUse' in content_block:
                        tool = content_block['toolUse']
                        if tool['name'] == 'top_song':
                            try:
                                song, artist = get_top_song(tool['input']['sign'])
                                tool_result = {
                                    "toolUseId": tool['toolUseId'],
                                    "content": [{"json": {"song": song, "artist": artist}}]
                                }
                            except ValueError as err:
                                tool_result = {
                                    "toolUseId": tool['toolUseId'],
                                    "content": [{"text": str(err)}],
                                    "status": 'error'
                                }
                        elif tool['name'] == 'calc':
                            
                            result = calc(tool['input']['operator'], tool['input']['operand1'], tool['input']['operand2'])
                            tool_result = {
                                "toolUseId": tool['toolUseId'],
                                "content": [{"json": {"result": result}}]
                            }

                        else:
                            raise ValueError(f"Tool {tool['name']} not supported.")
                        
                        tool_result_message = {
                                "role": "user",
                                "content": [{"toolResult": tool_result}]
                        }
                        messages.append(tool_result_message)

                        # Resend the messages including the tool result
                        response = bedrock.converse_stream(
                            modelId=model_id,
                            messages=messages,
                            system=system_prompts,
                            inferenceConfig=inference_config,
                            additionalModelRequestFields=additional_model_fields,
                            toolConfig=tool_config
                        )

                        # for chunk in response['stream']:
                        #     if 'contentBlockDelta' in chunk:
                        #         delta = chunk['contentBlockDelta']['delta']
                        #         if 'text' in delta:
                        #             yield delta['text']
                        #             text += delta['text']
                        
                        
                        tool_use = {}
                        text = ""
                        content = []
                                    
                        for chunk in response['stream']:
                            print("Chunk received:", chunk)  # Debug log to trace events

                            if 'messageStart' in chunk:
                                message = {'role': chunk['messageStart']['role']}
                            elif 'contentBlockStart' in chunk:
                                if 'toolUse' in chunk['contentBlockStart']['start']:
                                    tool = chunk['contentBlockStart']['start']['toolUse']
                                    tool_use['toolUseId'] = tool['toolUseId']
                                    tool_use['name'] = tool['name']
                            elif 'contentBlockDelta' in chunk:
                                delta = chunk['contentBlockDelta']['delta']
                                if 'toolUse' in delta:
                                    if 'input' not in tool_use:
                                        tool_use['input'] = ''
                                    tool_use['input'] += delta['toolUse']['input']
                                elif 'text' in delta:
                                    text += delta['text']
                                    yield delta['text']
                            elif 'contentBlockStop' in chunk:
                                if 'input' in tool_use:
                                    tool_use['input'] = json.loads(tool_use['input'])
                                    content.append({'toolUse': tool_use})
                                    tool_use = {}
                                else:
                                    content.append({'text': text})
                                    text = ''
                            elif 'messageStop' in chunk:
                                stop_reason = chunk['messageStop']['stopReason']
                                assistant_message = {'role': 'assistant', 'content': content}
                                messages.append(assistant_message)

            break

        if 'metadata' in chunk:
            metadata = chunk['metadata']
            if 'usage' in metadata:
                print("\nToken usage")
                print(f"Input tokens: {metadata['usage']['inputTokens']}")
                print(f"Output tokens: {metadata['usage']['outputTokens']}")
                print(f"Total tokens: {metadata['usage']['totalTokens']}")
            if 'metrics' in metadata:
                print(f"Latency: {metadata['metrics']['latencyMs']} milliseconds")

if __name__ == "__main__":
    for chunk in stream_conversation("What is the most popular song on WZPZ?"):
        print(chunk)
