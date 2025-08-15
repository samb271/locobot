from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from crossformer.model.crossformer_model import CrossFormerModel
import cv2
import numpy as np
import os
from enum import Enum

DISTANCE_THRESHOLD = 0.1
class Mode(Enum):
    NAV = 'nav'
    PICK = 'pick'
    PLACE = 'place'

# Load the model
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"":"cuda"}
)
crossformer = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")

def navigate():
    pass

def pick():
    pass

def place():
    pass

def detect(image, query):
    return moondream.point(image, query)["points"]

def reason(image, query):
    return moondream.query(image, query)["answer"]

def get_object_depth(depth_image, position):
    pass

def get_action(image, text_task, head_name="nav"):
    
    image = cv2.resize(image, (224, 224))  # shape becomes (224, 224, 3)
    image = image[np.newaxis, np.newaxis, ...]  # (1, 1, 224, 224, 3)
    
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.full((1, image.shape[1]), True, dtype=bool),
    }
    task = crossformer.create_tasks(texts=[text_task])
    return crossformer.sample_actions(observation, task, head_name=head_name, rng=1)
    

# Load your image
MODE = Mode.NAV
while True:
    image = Image.open("image.jpg")
    W, H = image.size
    
    points = detect(image, "croissants")
    if len(points) < 0:
        # we cannot detect the object, explore...
        pass
    x = int(points[0]['x']*W)
    y = int(points[0]['y']*H)
    
    distance_to_object = get_object_depth(depth_image=image, position=(x, y))
    
    if (distance_to_object < DISTANCE_THRESHOLD) and MODE==Mode.NAV:
        MODE=Mode.PICK
        while MODE==Mode.PICK:
            # we were exploring and got close enough to the object, time to pick it up...
            action = get_action(
                image,
                text_task='pick up the object identified by the red dot',
                head_name='single_arm'
            )
            # TODO: pass action to environment and get observation back...
            # CODE HERE
            
            pick_success = int(reason(
                image,
                query="Did the robot pick up the croissant? Answer with a bit: 1 for yes and 0 for no."
            ))
            if(pick_success):
                MODE=Mode.NAV
    
    # Copy image so original is untouched
    img_with_dots = np.array(image.copy())
    img_with_dots = cv2.cvtColor(img_with_dots, cv2.COLOR_RGB2BGR)
    
    # Draw dots with BGR red color
    for point in points:
        x = int(point['x'] * img_with_dots.shape[1])
        y = int(point['y'] * img_with_dots.shape[0])
        cv2.circle(img_with_dots, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    # Convert back to RGB
    img_with_dots = cv2.cvtColor(img_with_dots, cv2.COLOR_BGR2RGB)
    
    # Save using PIL instead of cv2.imwrite
    save_path = os.path.join(os.getcwd(), "croissant_detected.jpg")
    img_pil = Image.fromarray(img_with_dots)
    img_pil.save(save_path)
    print(f"Saved image with dots at {save_path}")
    
    image = cv2.resize(img_with_dots, (224, 224))  # shape becomes (224, 224, 3)
    
    # Convert from HWC to CHW format and add batch dimensions
    image = image[np.newaxis, np.newaxis, ...]  # (1, 1, 224, 224, 3)
    
    # TODO MRSS25: Implement navigation logic
    observation = {
        "image_primary": image,
        "timestep_pad_mask": np.full((1, image.shape[1]), True, dtype=bool),
    }
    task = crossformer.create_tasks(texts=["go to the object with the red dot on it"])
    action = crossformer.sample_actions(observation, task, head_name="nav", rng=1)
    print(action.shape)