import time
import threading
import traceback
import logging
import os
import random
from datetime import datetime
from flask import Flask, send_from_directory, jsonify
from concurrent.futures import ThreadPoolExecutor
from Config import Config
from Manager.DeliveryManager import DeliveryManager
from Communicator import Communicator
from Evaluation import Evaluator
from Navigation import Navigator
from typing import Optional

class LLMDelivery:
    def __init__(self, num_customers: Optional[int] = None,
                num_stores: Optional[int] = None,
                num_delivery_men: Optional[int] = None,
                input_folder: Optional[str] = "input",
                dataset_path: Optional[str] = None,
                visualization: bool = False,
                evaluation: bool = True,
                control_fn: callable = None,
                ):
        self.num_customers = num_customers if num_customers is not None else Config.CUSTOMER_NUM
        self.num_delivery_men = num_delivery_men if num_delivery_men is not None else Config.DELIVERY_MAN_NUM
        self.num_stores = num_stores if num_stores is not None else Config.STORE_NUM
        self.dt = Config.DT
        self.ue_update_dt = Config.UE_UPDATE_DT

        random.seed(Config.SEED)

        self.delivery_manager = DeliveryManager(
            num_customers=num_customers,
            num_stores=num_stores,
            num_delivery_men=num_delivery_men,
            input_folder=input_folder,
            dataset_path=dataset_path
        )
        self.communicator = None
        self.exit_event = threading.Event()
        self.visualization = visualization
        self.evaluation = evaluation
        self.evaluator = Evaluator(self.delivery_manager)
        self.navigator = None
        self.control_fn = control_fn

        self.init_logger()

        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_path, exist_ok=True)

        if self.visualization:
            static_folder = os.path.join(os.path.dirname(__file__), 'static')
            self.start_web_server(static_folder=static_folder)

    def start_web_server(self, port: int = 5001, static_folder: str = 'static'):
        """
        Start the Flask web server for frontend visualization
        """
        app = Flask(__name__, static_folder=static_folder)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/')
        def index():
            return send_from_directory(static_folder, 'index.html')

        @app.route('/data')
        def data():
            return jsonify(self.delivery_manager.to_json())

        server_thread = threading.Thread(
            target=lambda: app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
        )
        server_thread.daemon = True
        server_thread.start()

    def init_logger(self):
        # create logs directory
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # create log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'delivery_{timestamp}.log')

        # configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # filter out httpx logs
        logging.getLogger('httpx').setLevel(logging.INFO)

        # clear existing handlers
        root_logger.handlers = []

        # add file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)

        # add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)

    def init_communicator(self):
        self.communicator = Communicator(port=9000, ip='127.0.0.1', resolution=(720, 600))
        self.navigator = Navigator(self.communicator, self.exit_event, self.ue_update_dt)

    def spawn_delivery_manager(self):
        self.communicator.spawn_delivery_manager()

    def spawn_delivery_men(self):
        self.communicator.spawn_delivery_men(self.delivery_manager.delivery_men)

    def default_delivery_man_control_fn(self, delivery_man, delivery_manager, dt, communicator, exit_event, navigator):
        delivery_man.update_delivery_man(
            delivery_manager,
            dt,
            communicator,
            exit_event,
            navigator
        )

    def delivery_man_control_fn(self, delivery_man):
        if self.control_fn is None:
            self.default_delivery_man_control_fn(
                delivery_man,
                self.delivery_manager,
                self.dt,
                self.communicator,
                self.exit_event,
                self.navigator
            )
        else:
            self.control_fn(
                delivery_man,
                self.delivery_manager,
                self.dt,
                self.communicator,
                self.exit_event,
                self.navigator
            )

    def run(self):
        '''
        Run the simulation
        '''
        with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
            try:
                futures = []
                for delivery_man in self.delivery_manager.delivery_men:
                    # future = executor.submit(delivery_man.update_delivery_man, self.delivery_manager, self.dt, self.communicator, self.exit_event, self.navigator)
                    future = executor.submit(self.delivery_man_control_fn, delivery_man)

                    futures.append(future)

                step = 0
                while True:
                    # Check if all futures are done
                    if all(future.done() for future in futures):
                        self.logger.info("All delivery men have finished their tasks.")
                        break

                    if step == Config.MAX_STEPS:
                        self.logger.info(f"Step limit {Config.MAX_STEPS} reached, Simulation finished")
                        break

                    self.delivery_manager.create_orders()
                    self.update_physical_states()

                    if step % 100 == 0:
                        self.logger.info(f"Main Thread: Orders: {self.delivery_manager.orders}")

                    if self.evaluation:
                        if step % 100 == 0:
                            self.evaluator.evaluate(os.path.join(self.output_path, f"{time.time()}_{Config.get_evaluation_name()}_{step}.json"))
                            # self.logger.info("Evaluation finished")
                    step += 1

                    time.sleep(self.ue_update_dt)
            except KeyboardInterrupt:
                self.logger.info("Simulation interrupted by user")
                self.exit_event.set()
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                self.logger.error(traceback.format_exc())
                self.exit_event.set()
            finally:
                self.logger.info("Waiting for all delivery men to finish...")
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error in thread: {e}")

                self.logger.info("Simulation fully stopped.")

    def update_physical_states(self):
        delivery_man_ids = [delivery_man.id for delivery_man in self.delivery_manager.delivery_men]
        result = self.communicator.get_position_and_direction(delivery_man_ids)
        for object_id, values in result.items():
            position, direction = values
            self.delivery_manager.delivery_men[object_id].set_position(position)
            self.delivery_manager.delivery_men[object_id].set_direction(direction)

if __name__ == "__main__":
    llm_delivery = LLMDelivery(6, 4, 3, "input")
    llm_delivery.run()
