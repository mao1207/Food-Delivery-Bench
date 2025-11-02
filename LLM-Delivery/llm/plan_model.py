import json
from typing import List, Tuple, Set
import ast
import re

from Base.Types import Vector

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import init_chat_model
from langchain.llms import OpenAI  # or your custom LLM wrapper

# from pydantic import BaseModel, Field

# class Coordinate(BaseModel):
#     """
#     A coordinate with x and y components (floats).
#     """
#     x: float = Field(..., description="X coordinate.")
#     y: float = Field(..., description="Y coordinate.")

# class PointOnly(BaseModel):
#     next_waypoint: Coordinate = Field(..., description="next waypoint to go")

# class IdxOnly(BaseModel):
#     order_idx: int = Field(..., description="Index of an order in the delivery man's list.")

# class IdxPoint(BaseModel):
#     order_idx: int = Field(..., description="Index of an order in the delivery man's list.")
#     next_waypoint: Coordinate = Field(..., description="next waypoint to go.")

# class Speed(BaseModel):
#     new_speed: int = Field(..., description="The new speed (1..250).")

class LowerLevelAgent:
    """
    A class that holds a reference to a DeliveryMan and a DeliveryManager,
    and provides a tool-based LangChain agent. The agent can call these
    'tool' methods (like accept_order, pick_up_order) to manipulate the
    DeliveryMan's state.
    """
    def __init__(self, delivery_man, delivery_manager, model: str):
        """
        :param delivery_man: The DeliveryMan instance whose attributes we modify
        :param delivery_manager: The DeliveryManager instance for orders, allocations, etc.
        :param llm: The LLM used by the lower-level agent to decide which tool to call.
                    You can pass in a LangChain LLM wrapper or just do OpenAI(model="...")
        """
        self.delivery_man = delivery_man
        self.delivery_manager = delivery_manager
        # If no LLM is specified, we create a default one
        if "gpt" in model:
            self.llm = init_chat_model(model=model, model_provider="openai", temperature=0)
        # Build the list of LangChain tools from our local methods
        self.tools = self._build_tools()
        # Build an agent that can pick among these tools
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or another agent type
            verbose=True
        )

    # TODO: while loop to decide
    def decide(self, high_level_plan: str) -> str:
        """
        Feeds the high-level plan into the agent as a user prompt, letting
        the agent decide which tool to call. Returns the final text output
        from the agent (after it calls the tool).
        """
        result = self.agent.run(high_level_plan)
        print(result, flush=True)
        return result
    # -------------------------------------------------------------------
    # Build Tools
    # -------------------------------------------------------------------
    def _build_tools(self) -> List[Tool]:
        """
        Wrap each action method in a LangChain Tool object.
        The agent can call them by 'name' with arguments.
        """
        return [
            Tool(
                name="do_nothing",
                func=self.do_nothing,
                description="Do nothing this turn. Args: order_idx (int)."
            ),
            Tool(
                name="accept_order",
                func=self.accept_order,
                description="Accept an order. Args: order_idx (int)."
            ),
            Tool(
                name="pick_up_order",
                func=self.pick_up_order,
                description="Pick up an order. Args: order_idx (int), next_waypoint (Tuple)."
            ),
            Tool(
                name="deliver_order",
                func=self.deliver_order,
                description="Deliver an order. Args: order_idx (int), next_waypoint (Tuple)."
            ),
            Tool(
                name="buy_beverage",
                func=self.buy_beverage,
                description="Buy a beverage. Args: next_waypoint (Tuple)."
            ),
            Tool(
                name="open_shared_order",
                func=self.open_shared_order,
                description="Open a shared order. Args: order_idx (int), meeting_point (Tuple)."
            ),
            Tool(
                name="go_to_meeting_point",
                func=self.go_to_meeting_point,
                description="Go to shared order's meeting point. Args: order_idx (int)."
            ),
            Tool(
                name="cancel_order",
                func=self.cancel_order,
                description="Cancel a shared order. Args: order_idx (int)."
            ),
            Tool(
                name="change_speed",
                func=self.change_speed,
                description="Change the delivery man's speed. Args: new_speed (int)."
            ),
            # Tool(
            #     name="buy_bike",
            #     func=self.buy_bike,
            #     description="Buy a bike. No arguments for now."
            # ),
        ]
    # -------------------------------------------------------------------
    # Tools Implementation
    # (Each modifies self.delivery_man or calls delivery_manager)
    # -------------------------------------------------------------------
    def do_nothing(self, order_idx: str) -> str:
        """
        Args: order_idx (int)
        Do nothing, no args required
        """
        return "DeliveryMan does nothing this turn."

    def accept_order(self, idx: int) -> str:
        """
        Accept an unassigned order identified by 'order_idx'.

        Args:
            order_idx (int): Index of the order in the global list of orders (delivery_manager.get_orders())

        Return:
            A message describing the result of the attempt to accept.
        """
        # args = ast.literal_eval(args)
        idx = int(idx)
        all_orders = self.delivery_manager.get_orders()
        if 0 <= idx < len(all_orders):
            order = all_orders[idx]
            allocated = self.delivery_manager.allocate_order(order, self.delivery_man)
            if allocated:
                self.delivery_man.orders.append(order)
                return f"Order {idx} accepted."
            else:
                self.delivery_man.invalid_num += 1
                return f"Order {idx} could not be allocated."
        else:
            return f"Invalid order_idx={idx}."

    def pick_up_order(self, args: str) -> str:
        """
        Pick up an order (if not already picked up) and move to the specified 'next_waypoint'.
        Args:
            order_idx (int): Index of the order in the global list of orders (delivery_manager.get_orders())
            next_waypoint (Tuple): An integer representing the waypoint index to move to.

        Return:
            A message describing the action taken.
        """
        # data = json.loads(args_json)
        numbers = re.findall(r"-?\d+\.?\d*", args)
        numbers = [float(n) if '.' in n else int(n) for n in numbers]
        order_idx = numbers[0]
        waypoint = Vector(numbers[1], numbers[2])
        if 0 <= order_idx < len(self.delivery_man.orders):
            order = self.delivery_man.orders[order_idx]
            if not order.has_picked_up:
                self.delivery_man.current_order = order
                self.delivery_man.next_waypoint = waypoint
                self.delivery_man.state = self.delivery_man.State.MOVING
                return f"Moving to waypoint {waypoint} to pick up order {order_idx}."
            else:
                self.delivery_man.invalid_num += 1
                return f"Order {order_idx} already picked up."
        return f"Invalid order_idx={order_idx}."

    def deliver_order(self,args: str) -> str:
        """
        Deliver an order (if already picked up) to the specified 'next_waypoint'.
        Args:
            order_idx (int): Index of the order in the global list of orders (delivery_manager.get_orders())
            next_waypoint (Tuple): An integer representing the waypoint index to move to.
        Example:
            "order_idx": 2,
            "next_waypoint": 5
        Return:
            A message describing the delivery action taken (or if invalid).
        """
        numbers = re.findall(r"-?\d+\.?\d*", args)
        numbers = [float(n) if '.' in n else int(n) for n in numbers]
        order_idx = numbers[0]
        waypoint = Vector(numbers[1], numbers[2])
        if 0 <= order_idx < len(self.delivery_man.orders):
            order = self.delivery_man.orders[order_idx]
            if order.has_picked_up and not order.has_delivered:
                self.delivery_man.current_order = order
                self.delivery_man.next_waypoint = waypoint
                self.delivery_man.state = self.delivery_man.State.MOVING
                return f"Delivering order {order_idx} to waypoint {waypoint}."
            else:
                self.delivery_man.invalid_num += 1
                return f"Order {order_idx} not ready for delivery."
        return f"Invalid order_idx={order_idx}."

    def buy_beverage(self, args: str) -> str:
        """
        Move to the specified waypoint to buy a beverage, which may increase energy, etc.

        Args:
            next_waypoint (Tuple): An integer representing the waypoint index to move to which can buy a beverage.

        Return:
            A message describing the action taken.
        """
        numbers = re.findall(r"-?\d+\.?\d*", args)
        numbers = [float(n) if '.' in n else int(n) for n in numbers]
        wp = Vector(numbers[1], numbers[1])
        self.delivery_man.next_waypoint = wp
        self.delivery_man.state = self.delivery_man.State.MOVING
        return f"Moving to waypoint {wp} to buy beverage."

    def open_shared_order(self, args: str) -> str:
        """
        Make an existing order 'shared' and set its meeting point.

        Args:
            order_idx (int): Index of the order in the global list of orders (delivery_manager.get_orders())
            meeting_point (Tuple): An integer representing the meeting point.
        Return:
            A message describing the action.
        """
        numbers = re.findall(r"-?\d+\.?\d*", args)
        numbers = [float(n) if '.' in n else int(n) for n in numbers]
        order_idx = numbers[0]
        meeting_point = Vector(numbers[1], numbers[2])
        if 0 <= order_idx < len(self.delivery_man.orders):
            order = self.delivery_man.orders[order_idx]
            if not order.is_shared:
                order.is_shared = True
                order.meeting_point = meeting_point
                self.delivery_manager.add_shared_order(order)
                return f"Opened shared order {order_idx} with meeting point {meeting_point}."
            else:
                self.delivery_man.invalid_num += 1
                return f"Order {order_idx} is already shared."
        return f"Invalid order_idx={order_idx}."

    def go_to_meeting_point(self, order_idx: int) -> str:
        """
        Move to the shared order's meeting point (if the order is shared).
        Args:
            order_idx (int): An integer referencing an order in the delivery_man's list.
        Return:
            A message describing the action or if invalid.
        """
        idx = int(order_idx)
        if 0 <= idx < len(self.delivery_man.orders):
            order = self.delivery_man.orders[idx]
            if order.is_shared:
                self.delivery_man.next_waypoint = order.meeting_point
                self.delivery_man.current_order = order
                self.delivery_man.state = self.delivery_man.State.MOVING
                return f"Going to meeting point of order {idx}."
            else:
                self.delivery_man.invalid_num += 1
                return f"Order {idx} is not shared."
        return f"Invalid order_idx={idx}."

    def cancel_order(self, args: str) -> str:
        """
        Cancel a shared order (if it is shared) for this delivery man.
        Args:
            order_idx (int): The index of the order in this delivery_man's orders list.
        Return:
            A message describing the result of the cancellation attempt.
        """
        # numbers = re.findall(r"-?\d+\.?\d*", args)
        # numbers = [float(n) if '.' in n else int(n) for n in numbers]
        idx = int(args)
        if 0 <= idx < len(self.delivery_man.orders):
            order = self.delivery_man.orders[idx]
            if order.is_shared:
                canceled = self.delivery_manager.cancel_shared_order(order)
                if canceled:
                    order.is_shared = False
                    order.meeting_point = None
                    return f"Shared order {idx} canceled."
                else:
                    self.delivery_man.invalid_num += 1
                    return f"Could not cancel shared order {idx}."
            else:
                return f"Order {idx} is not shared."
        return f"Invalid order_idx={idx}."

    def change_speed(self, new_speed_str: int) -> str:
        """
        Change the DeliveryMan's move speed (cm/s).
        Args:
            new_speed_str (int): The new speed (int), must be between 1 and 250.
        Return:
            A message describing the action or if invalid.
        """
        new_speed = int(new_speed_str)
        if 0 < new_speed <= 250:
            self.delivery_man.set_speed(new_speed)
            return f"Speed changed to {new_speed}."
        else:
            self.delivery_man.invalid_num += 1
            return f"Invalid speed {new_speed}."
    # def buy_bike(self) -> str:
    #     # Example "buy bike" logic
    #     return "Bought a bike (logic not implemented)."