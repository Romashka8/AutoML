# --------------------------------------------------------------------------------------------------------------

import abc

# --------------------------------------------------------------------------------------------------------------

class Parameters(abc.ABC):

	"""
	Class for parameters controling.
	Set model parameters in hyperopt style.
	"""

	@abc.abstractmethod
	def __init__(self, **params_dict):

		"""
		Init model parameters.
		"""

		pass

	@abc.abstractmethod
	def param_transformer(self, **params_dict):

		"""
		Transform parameters to valid hyperopt type.
		"""

# --------------------------------------------------------------------------------------------------------------

class TrainModel(abc.ABC):

	"""
	Automized model training pipeline.
	"""

	@abc.abstractmethod
	def __init__(self, **params_dict):

		"""
		Init samples and model parameters.
		"""

		pass

	@abc.abstractmethod
	def train(self, **params_dict):

		"""
		Train model implementation.
		"""

		pass

# --------------------------------------------------------------------------------------------------------------

class FeatureSelector(abc.ABC):

	"""
	Feature selector implementstion.
	"""

	@abc.abstractmethod
	def __init__(self, **params_dict):

		"""
		Init samples and other parameters.
		"""

		pass

	@abc.abstractmethod
	def get_feature_importance(self):

		"""
		Return model feature importances.
		"""

		pass

	@abc.ABC
	def get_scores(self):

		"""
		Return model scores.
		"""

		pass

# --------------------------------------------------------------------------------------------------------------
