## modules() vs children()

model.modules():
* This function recursively goes through all the modules in the model, including the model itself, its sub-modules, and their sub-modules, forming a flattened structure. <br>
* It returns an iterator over all the modules, including the model itself and all its sub-modules in a depth-first manner. <br>
