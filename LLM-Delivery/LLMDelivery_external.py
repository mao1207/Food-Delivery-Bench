import base64
import io
import json
import logging
import threading
import time
import traceback
# from typing import List, Tuple

from Base.DeliveryMan import DeliveryMan
from Base.DeliveryManState import DeliveryManState, DeliveryManPhysicalState, DeliveryManDrivingState, DeliveryManWalkingState
from Base.ActionSpace import Action, ActionSpace
from Communicator import Communicator
from Config import Config
from Evaluation import Evaluator
from LLMDelivery import LLMDelivery
from Manager.DeliveryManager import DeliveryManager
from Navigation import Navigator
from llm.openai_model import UEOpenAIModel


delivery_man_system_prompt = """You are an agent in a simulated world who can navigate freely in the environment using JSON blobs. You will be given a task, and you should try to solve it as best as you can.
To do so, you have been given access to a list of tools: these tools are Python functions which you can call to perform actions in the environment. You can call these functions by providing the function name and arguments in a JSON format.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Observation' -> 'Action' sequence.

At each step, the 'Observation' is provided to you by the user as text and/or image.
Then, in the 'Action' sequence, you should provide the function calls that you want to use. The way you use the tools is by providing a JSON array of objects.
"""

# delivery_man_user_prompt = """Task: Walk towards the nearest street light.

# """

# delivery_man_user_prompt = """Task: You are standing on a sidewalk. There is a box at the other end of the side walk in the front. Your goal is to reach as close to the box as possible.

# """

delivery_man_user_prompt = """Task: You are standing on a sidewalk. There is a box at the other end of the side walk that you are standing at.
Your goal is to reach as close to the box as possible in the shortest amount of time. The more time you take, the more penalty you will receive.

"""


delivery_man_observation_prompt = """Observation:
- Current position of agent (in cm units): {position}
- Movement speed of the agent (in cm/s): {speed}
- Current view of the agent (base64 enocoded): attached
"""

delivery_man_task_complete_observation = """- is_task_complete(): {task_status}"""

# GPT-3 tool description format
# delivery_man_tools = [
#     {
#         "type": "function",
#         "name": "move_forward",
#         "description": "Move agent forward for input seconds",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "dt": {
#                     "type": "float",
#                     "description": "Seconds to move forward",
#                 }
#             },
#             "required": [
#                 "dt"
#             ],
#             "additionalProperties": False
#         }
#     },
#     {
#         "type": "function",
#         "name": "rotate_right",
#         "description": "Rotate the agent in right direction",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "angle": {
#                     "type": "float",
#                     "description": "Angle in degrees",
#                 }
#             },
#             "required": [
#                 "angle"
#             ],
#             "additionalProperties": False
#         }
#     },
#     {
#         "type": "function",
#         "name": "rotate_left",
#         "description": "Rotate the agent in left direction",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "angle": {
#                     "type": "float",
#                     "description": "Angle in degrees",
#                 }
#             },
#             "required": [
#                 "angle"
#             ],
#             "additionalProperties": False
#         }
#     },
# ]

delivery_man_tools = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move agent forward for input seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dt": {
                        "type": "number",
                        "minimum": 0.01,
                        "maximum": 300.0,       # 5 minutes
                        "description": "Seconds to move forward",
                    }
                },
                "required": [
                    "dt"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Rotate the agent in left direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 360.0,
                        "description": "Angle in degrees.",
                    }
                },
                "required": [
                    "angle"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "Rotate the agent in right direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 360.0,
                        "description": "Angle in degrees.",
                    }
                },
                "required": [
                    "angle"
                ],
                "additionalProperties": False
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "is_task_complete",
    #         "description": "Check if the task is complete.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "angle": {
    #                     "type": "number",
    #                     "minimum": 0.0,
    #                     "maximum": 360.0,
    #                     "description": "Angle in degrees.",
    #                 }
    #             },
    #             "required": [
    #                 "angle"
    #             ],
    #             "additionalProperties": False
    #         }
    #     }
    # },
]


def function_calling(llm: UEOpenAIModel, messages: dict, functions: list, retry=5):
    max_tokens = 2048

    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        max_tokens=max_tokens,
        # temperature=temperature,
        # top_p=top_p,
        # n=num_return_sequences,
        # tools=functions if functions else [],
        tools=functions,
        tool_choice="required"
    )

    # print("LLM response")
    # print(response)
    # ChatCompletion(
    #   id='chatcmpl-BNZSZVzZa9KlMxUJEjNh3s0hDJbqW',
    #   choices=[
    #       Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_98kDW2YGAXCHpjhSbFnhWqQG', function=Function(arguments='{"dt":10}', name='move_forward'), type='function')], annotations=[]))], created=1744957551, model='gpt-4o-2024-11-20', object='chat.completion', service_tier='default', system_fingerprint='fp_3bdddbcbe8', usage=CompletionUsage(completion_tokens=15, prompt_tokens=826, total_tokens=841, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))
    # )

    # process the returned results
    results = []
    for choice in response.choices:
        if hasattr(choice, 'function') and choice.function:
            results.append({
                "function_call": {
                    "name": choice.function.name,
                    "arguments": choice.function.arguments
                }
            })
        else:
            results.append(choice.message.tool_calls)

    return results

def get_user_message(position, speed, view, first_message):
    obsv_prompt = delivery_man_observation_prompt.format(
        position=(position.x, position.y),
        speed=speed,
    )
    if first_message:
        user_text = delivery_man_user_prompt + obsv_prompt
    else:
        user_text = obsv_prompt

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{view}"
                }
            }
        ]
    }


def convert_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def move_forward_fn(delivery_man: DeliveryMan, communicator: Communicator, dt: float):
    delivery_man.logger.info(f"Moving forward for {dt} seconds")
    communicator.delivery_man_move_forward(delivery_man.id)
    time.sleep(dt)
    communicator.delivery_man_stop(delivery_man.id)


def turn_left_fn(delivery_man: DeliveryMan, communicator: Communicator, angle: float):
    delivery_man.logger.info(f"Turning left for {angle} degrees")
    communicator.delivery_man_rotate(delivery_man.id, angle, "left")


def turn_right_fn(delivery_man: DeliveryMan, communicator: Communicator, angle: float):
    delivery_man.logger.info(f"Turning right for {angle} degrees")
    communicator.delivery_man_rotate(delivery_man.id, angle, "right")


def external_control_fn(
        delivery_man: DeliveryMan,
        delivery_manager: DeliveryManager,
        dt: int,
        communicator: Communicator,
        exit_event: threading.Event,
        navigator: Navigator,
):
    # For testing purposes, use similar to default_fn
    # This function should work exactly the same way.
    # delivery_man.update_delivery_man(
    #     delivery_manager,
    #     dt,
    #     communicator,
    #     exit_event,
    #     navigator
    # )

    function_name_to_fn_mapping = {
        "move_forward": lambda dt: move_forward_fn(delivery_man, communicator, dt),
        "turn_left": lambda angle: turn_left_fn(delivery_man, communicator, angle),
        "turn_right": lambda angle: turn_right_fn(delivery_man, communicator, angle),
    }

    logger: logging.Logger = delivery_man.logger

    logger.info(f"External Thread for DeliveryMan {delivery_man.id} started")
    steps = 0

    messages = [
        {"role": "system", "content": delivery_man_system_prompt},
    ]

    start_pos = (1700, -18300)      # 110 rotation
    street_light = (3320, -18810)
    # target_vector = delivery_man.get_position().__class__(3050, -18750)
    box = (16646.52, -18311.55)
    target_vector = delivery_man.get_position().__class__(16450, -18300)

    try:
        while delivery_man.get_position().distance(target_vector) > 200 and steps < 100:
            steps += 1
            position = delivery_man.get_position()
            speed = delivery_man.get_speed()
            view = delivery_man.get_current_view(communicator)
            view_base64 = convert_image_to_base64(view)

            messages.append(get_user_message(position, speed, view_base64, first_message = (steps == 1)))

            llm_result = function_calling(delivery_man.llm, messages, delivery_man_tools)
            # logger.info(f"LLM results: ({len(llm_result)}, {len(llm_result[0])})")
            # [[ChatCompletionMessageToolCall(id='call_98kDW2YGAXCHpjhSbFnhWqQG', function=Function(arguments='{"dt":10}', name='move_forward'), type='function')]]

            # logger.info(llm_result)
            # logger.info(llm_result[0][0].function)
            # logger.info(llm_result[0][0].function.name, llm_result[0][0].function.arguments)

            if llm_result[0][0].function.name == "move_forward":
                dt = json.loads(llm_result[0][0].function.arguments)["dt"]
                function_name_to_fn_mapping["move_forward"](dt)
            elif llm_result[0][0].function.name == "turn_left":
                angle = json.loads(llm_result[0][0].function.arguments)["angle"]
                function_name_to_fn_mapping["turn_left"](angle)
            elif llm_result[0][0].function.name == "turn_right":
                angle = json.loads(llm_result[0][0].function.arguments)["angle"]
                function_name_to_fn_mapping["turn_right"](angle)
            else:
                logger.warn(f"Unknown function name: {llm_result[0][0].function}")

            # if steps % 5 == 0:
            logger.info(f"DeliveryMan({delivery_man.id}) Steps={steps}, Position={position}")

    except Exception as e:
        delivery_man.logger.error(f"Error in DeliveryMan {delivery_man.id}: {e}")
        delivery_man.logger.error(traceback.format_exc())
        delivery_man.logger.error(f"Thread for DeliveryMan {delivery_man.id} is dead")

