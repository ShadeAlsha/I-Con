class EMAUtils:
    """Utility class for EMA operations."""
    
    @staticmethod
    def create_ema_copy(model: nn.Module) -> nn.Module:
        """Create an EMA copy of a model."""
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @staticmethod
    def update_ema(ema_model: nn.Module, model: nn.Module, momentum: float) -> None:
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data = momentum * ema_param.data + (1.0 - momentum) * param.data