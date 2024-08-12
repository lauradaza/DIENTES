import clip
import torch

from src.utils.constants import ORGAN_NAMES
from src.utils.constants import TASK_MODALITIES, TEMPLATE


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device, download_root="./pretrained_weights")

PROMPTS = ["A {modality} of a {item}"]

temp_prompts = []  # keep track of the items/modalities so far
prompt_inputs = []  # modality and item to use in the clip prompts
task_indexes = {}  # which promtps belong to which task
for task in TASK_MODALITIES:
    task_list = []
    for organ in TEMPLATE[task]:
        prompt = PROMPTS[0].format(
            modality=TASK_MODALITIES[task], item=ORGAN_NAMES[organ].lower()
        )

        if prompt in temp_prompts:
            task_list.append(temp_prompts.index(prompt))
            continue

        temp_prompts.append(prompt)
        task_list.append(temp_prompts.index(prompt))
        prompt_inputs.append([TASK_MODALITIES[task], ORGAN_NAMES[organ]])
    task_indexes[task] = task_list

# Calculate text embedding features
all_text_features = []
for pt in PROMPTS:
    text_inputs = torch.cat(
        [
            clip.tokenize(pt.format(modality=values[0], item=values[1]))
            for values in prompt_inputs
        ]
    ).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        all_text_features.append(text_features)
        print(text_features.shape, text_features.dtype)

all_text_features = torch.stack(all_text_features).mean(0)
print(all_text_features.shape, all_text_features.dtype)

output = {
    "task_indexes": task_indexes,
    "text_features": text_features,
}
torch.save(output, "pretrained_weights/txt_encoding_TF3.pth")
