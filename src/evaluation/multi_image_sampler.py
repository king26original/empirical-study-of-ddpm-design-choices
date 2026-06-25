import torch
from accelerate import Accelerator

accelerator=Accelerator(mixed_precision="fp16")

def sample_batch(
    ema_model,
    alpha,
    alpha_bar,
    posterior_variance,
    batch_size=64
):
    ema_model.eval()
    ema_model=accelerator.prepare(ema_model)
    alpha = alpha.to(accelerator.device)
    alpha_bar = alpha_bar.to(accelerator.device)
    posterior_variance = posterior_variance.to(accelerator.device)

    xt = torch.randn(
        (batch_size, 3, 32, 32),
        device=accelerator.device
    )

    with torch.no_grad():
        for t in reversed(range(1000)):
            t_tensor = torch.full(
                (batch_size,),
                t,
                device=accelerator.device,
                dtype=torch.long
            )

            e_pred = ema_model(xt, t_tensor)

            if t > 0:
                z = torch.randn_like(xt)
            else:
                z = torch.zeros_like(xt)

            xt = (
                (1 / torch.sqrt(alpha[t]))
                * (
                    xt
                    - ((1 - alpha[t])
                    / torch.sqrt(1 - alpha_bar[t]))
                    * e_pred
                )
                + torch.sqrt(posterior_variance[t]) * z
            )

    return xt
