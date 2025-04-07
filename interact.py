import importlib
import re

import config
from predict import predict_emotion
from train_model import get_main_outcome

# Create a single config instance that will be shared across all modules
single_config = config.Config()


def edit_config_file(num_epochs=None, learning_rate=None, batch_size=None):
    """Update configuration parameters in both memory and config.py file"""
    updates = {}
    if num_epochs is not None:
        updates['NUM_EPOCHS'] = num_epochs
    if learning_rate is not None:
        updates['LEARNING_RATE'] = learning_rate
    if batch_size is not None:
        updates['BATCH_SIZE'] = batch_size

    if updates:
        # Update in memory
        single_config.update(**updates)

        # Update config.py file
        with open('config.py', 'r') as file:
            content = file.read()

        for key, value in updates.items():
            # Find the line with the config value and update it
            pattern = rf'self\.{key}\s*=\s*[^#\n]+'
            if isinstance(value, str):
                new_line = f'self.{key} = "{value}"'
            else:
                new_line = f'self.{key} = {value}'
            content = re.sub(pattern, new_line, content)

        with open('config.py', 'w') as file:
            file.write(content)

        print(f"[*] Updated config: {updates}")
        print("[*] Changes have been written to config.py")
