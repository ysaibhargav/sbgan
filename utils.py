import yaml

def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd)
    return data

class AttributeDict(dict):
	def __getattr__(self, attr):
		return self[attr]
	def __setattr__(self, attr, value):
		self[attr] = value
	def __hash__(self):
		return hash(tuple(sorted(self.items())))