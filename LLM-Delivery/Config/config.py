class Config:
    SEED = 1
    DT = 1
    UE_UPDATE_DT = 0.1

    SIDEWALK_OFFSET = 1700  # distance between the sidewalk and the road center
    NUM_THREADS = 20

    # DELIVERY
    MAX_ORDERS = 3
    PROBABILITY_TO_CREATE_NEW_ORDER = 0.8
    COST_OF_BEVERAGE = 5
    DELIVERY_MAN_RECOVER_ENERGY_AMOUNT = 60
    DELIVERY_MAN_MEET_DISTANCE = 500
    PRICE_OF_BIKE = 100
    DIFFICULTY = 'medium'  # easy, medium, hard
    MAX_STEPS = 10000
    STORE_NUM = 6
    CUSTOMER_NUM = 4
    DELIVERY_MAN_NUM = 3

    # DELIVERY MAN
    DELIVERY_MAN_WALK_ARRIVE_WAYPOINT_DISTANCE = 200
    DELIVERY_MAN_DRIVE_ARRIVE_WAYPOINT_DISTANCE = 400
    DELIVERY_MAN_INITIAL_ENERGY = 100
    DELIVERY_MAN_MIN_SPEED = 200  # unit: cm/s
    DELIVERY_MAN_MAX_SPEED = 350  # unit: cm/s
    USE_A2A_PLANNER = False
    MODEL_TYPE = "base"
    MODEL_NAME = "google/gemini-2.0-flash-001"

    # Navigation
    PID_KP = 0.15
    PID_KI = 0.005
    PID_KD = 0.12

    # UE
    DELIVERY_MAN_MODEL_PATH = "/Game/TrafficSystem/Pedestrian/BP_DeliveryMan.BP_DeliveryMan_C"
    DELIVERY_MANAGER_MODEL_PATH = "/Game/TrafficSystem/DeliveryManager.DeliveryManager_C"
    SCOOTER_MODEL_PATH = "/Game/ScooterAssets/Blueprints/BP_Scooter_Pawn.BP_Scooter_Pawn_C"

    def get_evaluation_name(self):
        return f"{self.DIFFICULTY}_{self.MODEL_NAME}_{self.STORE_NUM}_{self.CUSTOMER_NUM}_{self.DELIVERY_MAN_NUM}"
