# N801

## Introduction
This is a code workspace for audio deepfake. it has collected a lot of model be apply on deepfake detection, especially those SOTA model.

The detail of repo is as follows:
1. Origin version of SOTA model, not hype parameters are changed. this part is located in `model/collection`
2. finetune version of SOTA model, the parameters are changed to finetune to new dataset, the model configuration not guarantee to got best performance. this part is located in `model/finetune`
3. some useful speech module , such as loss function, model architecture and MOS predictor.locate in `speech_tool`
4. idea for audio deepfake. locate in `idea`

The final purpose is convenient for competition and experiment.
## Model

Each direction under `model/` should include three component: 
1. `README.md` : to record some information about this model.
2. `config.yaml`： to record some train configuration.
3. `model.py`: the model definition python file.