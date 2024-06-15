# GAI HW4

## Background
This work combines Deep Image Prior (DIP) with DDPM-inspired supervision. By observing the PSNR values at each stage, we can determine the optimal stopping point based on when the PSNR is the highest.

## How to Work
In `main.ipynb`, there is a **Setting Parameters** block where you can configure the following parameters:

| Name         | Description                                                  |
|--------------|--------------------------------------------------------------|
| `with_ddpm`  | Set to `True` for DIP with DDPM, `False` for traditional DIP |
| `noisy_beta` | The noise level for images                                   |
| `num_iter`   | The total number of steps for running DIP                    |
| `num_stages` | The total number of stages for DDPM                          |
| `max_beta`   | The maximum noise level for DDPM                             |
| `model_name` | Choose a model to use: `skip`, `unet`, or `cnn`              |

Configure these parameters to tailor the experiment to your needs and observe the effects on the PSNR values and early stopping criteria.

### Example File
Example filenames for reference: `project4_{with_ddpm}_{num_stages}_{model_name}_{noisy_beta}_{max_beta}_{num_iter}.ipynb`