
def countTrainableParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def countAllParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    return num_params