from pathlib import Path
import json
import yaml
import toml


class PropertyUtils:
    @staticmethod
    def load_property_file(properties_file):
        if isinstance(properties_file, str):
            properties_file = Path(properties_file)

        with open(properties_file) as fp:
            if properties_file.suffix == ".json":
                properties = json.load(fp)

            elif properties_file.suffix == ".yaml":
                properties = yaml.load(fp, Loader=yaml.SafeLoader)

            elif properties_file.suffix == ".toml":
                properties = toml.load(fp)
            else:
                raise ValueError(f'Error loading {properties_file.name}. {properties_file.suffix} is not supported')

        return properties
