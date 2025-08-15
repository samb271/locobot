from enum import Enum
import cv2
import numpy as np
from PIL import Image
from crossformer.model.crossformer_model import CrossFormerModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"":"cuda"}
)
crossformer = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")

class Mode(Enum):
    NAV_TO_OBJECT = 1
    PICK = 2
    NAV_TO_BASKET = 3
    PLACE = 4
    COMPLETE = 5

class TaskState:
    def __init__(self):
        self.target_object = "croissants"
        self.receiving_object = "basket"
        self.object_picked = False
        self.basket_found = False
        self.exploration_counter = 0
        self.max_exploration_steps = 50
        
def detect(image, query):
    image = Image.fromarray(image)
    return moondream.point(image, query)["points"]


def get_action(image, text_task, head_name="nav"):
    
    image = cv2.resize(image, (224, 224))  # shape becomes (224, 224, 3)
    image = image[np.newaxis, np.newaxis, ...]  # (1, 1, 224, 224, 3)
    
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.full((1, image.shape[1]), True, dtype=bool),
    }
    task = crossformer.create_tasks(texts=[text_task])
    return crossformer.sample_actions(observation, task, head_name=head_name, rng=1)
    

def pick(image, text_task, head_name="single_arm"):
    
    image = cv2.resize(image, (224, 224))  # shape becomes (224, 224, 3)
    image = image[np.newaxis, np.newaxis, ...]  # (1, 1, 224, 224, 3)
    
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.full((1, image.shape[1]), True, dtype=bool),
    }
    task = crossformer.create_tasks(texts=[text_task])
    return crossformer.sample_actions(observation, task, head_name=head_name, rng=1)

def place(image, text_task, head_name="single_arm"):
    
    image = cv2.resize(image, (224, 224))  # shape becomes (224, 224, 3)
    image = image[np.newaxis, np.newaxis, ...]  # (1, 1, 224, 224, 3)
    
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.full((1, image.shape[1]), True, dtype=bool),
    }
    task = crossformer.create_tasks(texts=[text_task])
    return crossformer.sample_actions(observation, task, head_name=head_name, rng=1)

def rotate_and_search(angle=30):
    """Rotate robot to explore environment"""
    # TODO: Rotate robot by specified angle
    # robot.rotate(angle)
    pass

def get_object_depth(depth_image, position):
    """Extract depth value at given pixel position"""
    x, y = position
    if hasattr(depth_image, 'shape') and len(depth_image.shape) == 3:
        # Convert RGB to depth if needed, or load actual depth image
        return np.random.uniform(0.5, 3.0)  # Placeholder
    return depth_image[y, x] if depth_image is not None else 1.0

def execute_action(action):
    """Pass crossformer action to env"""
    # TODO: Interpret action tensor and send to robot
    pass

def reason(image, query):
    image = Image.fromarray(image)
    return moondream.query(image, query)["answer"]

def add_dot(image, x, y):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    x = int(x * image.shape[1])
    y = int(y * image.shape[0])
    cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main_algorithm():
    # Constants
    DISTANCE_THRESHOLD = 0.2  # meters
    
    # Initialize state
    state = TaskState()
    mode = Mode.NAV_TO_OBJECT
    
    print(f"Starting task: Pick up {state.target_object} and place in {state.receiving_object}")
    
    while mode != Mode.COMPLETE:
        # Get current observation
        image = Image.open("image.jpg")  # TODO: Get from robot camera
        depth_image = None  # TODO: Get from robot depth sensor
        W, H = image.size
        image = np.array(image.copy())
        
        if mode == Mode.NAV_TO_OBJECT:
            print("Phase 1: Navigating to target object...")
            
            # Detect target object
            points = detect(image, state.target_object)
            
            if len(points) == 0:
                # Object not visible, explore environment
                state.exploration_counter += 1
                if state.exploration_counter > state.max_exploration_steps:
                    print("Failed to find target object after exploration")
                    break
                
                print("Object not detected, exploring...")
                rotate_and_search()
                continue
            
            # Object detected, navigate towards it
            x = int(points[0]['x'] * W)
            y = int(points[0]['y'] * H)
            distance_to_object = get_object_depth(depth_image, (x, y))
            image = add_dot(image, x, y)
            
            print(f"Object detected at distance: {distance_to_object:.2f}m")
            
            if distance_to_object < DISTANCE_THRESHOLD:
                # Close enough to attempt picking
                mode = Mode.PICK
                print("Switching to PICK mode")
            else:
                # Navigate closer to object
                nav_action = get_action(
                    image, 
                    text_task=f'navigate to the {state.target_object}',
                    head_name='nav'
                )
                execute_action(nav_action)
        
        elif mode == Mode.PICK:
            print("Phase 2: Picking up object...")
            
            # Generate pick action
            pick_action = get_action(
                image,
                text_task=f'pick up the {state.target_object}',
                head_name='single_arm'
            )
            
            # Execute pick action
            execute_action(pick_action)
            
            # Verify pick success
            pick_success = int(reason(
                image,
                query=f"Did the robot successfully pick up the {state.target_object}? Answer with 1 for yes, 0 for no."
            ))
            
            if pick_success:
                state.object_picked = True
                mode = Mode.NAV_TO_BASKET
                print("Pick successful! Switching to NAV_TO_BASKET mode")
            else:
                print("Pick failed, retrying...")
                # Could add retry logic or repositioning here
        
        elif mode == Mode.NAV_TO_BASKET:
            print("Phase 3: Navigating to receiving object...")
            
            # Detect receiving object (basket)
            basket_points = detect(image, state.receiving_object)
            
            if len(basket_points) == 0:
                # Basket not visible, explore environment
                print("Basket not detected, exploring...")
                rotate_and_search()
                continue
            
            # Basket detected, navigate towards it
            basket_x = int(basket_points[0]['x'] * W)
            basket_y = int(basket_points[0]['y'] * H)
            distance_to_basket = get_object_depth(depth_image, (basket_x, basket_y))
            
            print(f"Basket detected at distance: {distance_to_basket:.2f}m")
            
            if distance_to_basket < DISTANCE_THRESHOLD:
                # Close enough to attempt placing
                mode = Mode.PLACE
                print("Switching to PLACE mode")
            else:
                # Navigate closer to basket
                nav_action = get_action(
                    image,
                    text_task=f'navigate to the {state.receiving_object}',
                    head_name='nav'
                )
                execute_action(nav_action)
        
        elif mode == Mode.PLACE:
            print("Phase 4: Placing object...")
            
            # Generate place action
            place_action = get_action(
                image,
                text_task=f'place the {state.target_object} in the {state.receiving_object}',
                head_name='single_arm'
            )
            
            # Execute place action
            execute_action(place_action)
            
            # Verify place success
            place_success = int(reason(
                image,
                query=f"Did the robot successfully place the {state.target_object} in the {state.receiving_object}? Answer with 1 for yes, 0 for no."
            ))
            
            if place_success:
                mode = Mode.COMPLETE
                print("Task completed successfully!")
            else:
                print("Place failed, retrying...")
                # Could add retry logic here
    
    print("Algorithm finished")

if __name__ == "__main__":
    main_algorithm()