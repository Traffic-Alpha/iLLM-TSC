# Human-like Assessment of RL Actions: LLM's Role in TSC Optimization

## Info
We propose a framework that utilizes LLM to support RL models. This framework refines RL decisions based on real-world contexts and provides reasonable actions when RL agents make erroneous decisions. 

<div align=center>
<img width="90%" src="./images/RL_LLM_Framework.png" />
The detailed structure of HARLA.
</div>

## Typical Cases

- Case1: LLM think that the action taken by the RL Agent was unreasonable and gave a reasonable explanation and recommended actions.
<div align=center>
<img width="90%" src="./images/Case1.png" />


</div>

- Case 2: LLM considers that the movement made by the RL Agent is not the movement with the highest current mean occupancy but it is reasonable, after which LLM gives an explanation and recommendation.
<div align=center>
<img width="90%" src="./images/Case2.png" />
</div>

- Case 3: An ambulance needs to pass through the intersection, but the RL Agent does not take into account that the ambulance needs to be prioritized. LLM modifies the RL Agent’s action to prioritize the ambulance to pass through the intersection.
<div align=center>
<img width="90%" src="./images/Case3.png" />
</div>

## Run Evaluation


```bash
git clone https://github.com/pkunlp-icler/PCA-EVAL.git
cd TSC-HARLA
```

### RL Model Training

```bash
python sb3_ppo.py
```
### RL+LLM

```bash
python rl_llm_tsc.py
```

**Evaluation Rule: To make fair evaluation and comparison among different models, make sure you use the same LLM evaluation model (we use GPT4) for all the models you want to evaluate. Using a different scoring model or API updating might lead to different results.**