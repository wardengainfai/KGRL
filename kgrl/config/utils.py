from kgrl.config.base.ExperimentConfig import Config, ExperimentConfig
from typing import Dict, List, Any, Tuple, Union


def force_new_parameters_for_config(config: Config, force_parameters: Dict[str, Any]) -> Tuple[Config, Dict[str, Any]]:
	"""
	Change the parameters in the given config by those parameters given in
		force_parameters.

	Returns the changed config and a dictionary of unmapped items (those that
		could not be found in the config).
	"""
	force_parameters_list = list(force_parameters.items())
	for parameter, new_value in force_parameters_list:
		if parameter in config.__dict__.keys():
			# todo: check and constrain to type annotations
			config.__dict__[parameter] = new_value
			force_parameters.pop(parameter, None)
		else:
			for config_parameter, value in config.__dict__.items():
				if isinstance(value, Config):
					config.__dict__[config_parameter], force_parameters = force_new_parameters_for_config(value, force_parameters)
					force_parameters_list = force_parameters.items()
	return config, force_parameters

def group_configs_on_key(
		configs: List[Config],
		key: str
) -> Dict[Union[str, int, float], List[Config]]:
	"""
	Create a list of Configs. grouping all the given Configs with the same
	value in the given key.
	"""
	new_config_dict = {}

	for config in configs:
		# search for key in the config
		value = get_value(config, key)
		if value is not None:
			if new_config_dict.get(str(value)) is not None:
				new_config_dict[str(value)].append(config)
			else:
				new_config_dict[str(value)] = [config]
		else:
			print(f"Key: {key} not in the configs")
			break
	return new_config_dict


def get_value(config: Config, key: str) -> Union[int, str, None]:
	"""
	Get the value corresponding to a key in the given Config. Returns None if the
	key has not been found in the config.
	"""
	if key in config.__dict__.keys():
		value = config.__dict__[key]
		return value
	else:  # iterate down the subconfigs
		for config_parametter, value in config.__dict__.items():
			if isinstance(value, Config):
				value = get_value(value, key)
				if value is not None:
					return value
