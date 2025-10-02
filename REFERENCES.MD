## Chapter 1: Why model rearchitecting matters

Bhattacharyya, C., & Kim, Y. (2025). FineScope : Precision Pruning for Domain-Specialized Large Language Models Using SAE-Guided Self-Data Cultivation. http://arxiv.org/abs/2505.00624
* Finescope is used as an example of the efficiency that can be achieved by using a data-driven re-architecture.

Sreenivas, S. T., Muralidharan, S., Joshi, R., Chochowski, M., Mahabaleshwarkar, A. S., Shen, G., Zeng, J., Chen, Z., Suhara, Y., Diao, S., Yu, C., Chen, W.-C., Ross, H., Olabiyi, O., Aithal, A., Kuchaiev, O., Korzekwa, D., Molchanov, P., Patwary, M., … Catanzaro, B. (2024). LLM Pruning and Distillation in Practice: The Minitron Approach. http://arxiv.org/abs/2408.11796
* NVIDIA's framework for creating its models is based on the inspiration from the tailoring pipeline flow presented in the chapter.

## Chapter 2: Rearchitecting a LLM
Kim, B.-K., Kim, G., Kim, T.-H., Castells, T., Choi, S., Shin, J., & Song, H.-K. (2024). Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods. http://arxiv.org/abs/2402.02834
* Although it's an introductory chapter to the depth pruning technique, part of the explanations and methodology are based on the paper's findings.

## Chapter 3: Dissecting modern transformers. 
Tyukin, G., Dovonon, G. J.-S., Kaddour, J., & Minervini, P. (2024). Attention Is All You Need But You Don’t Need All Of It For Inference of Large Language Models. http://arxiv.org/abs/2407.15516
* The original paper. It introduced the Transformer architecture and the self-attention mechanism.

Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. http://arxiv.org/abs/1911.02150
* MQA was proposed in 2019, in 2024 it's the architecture that Google uses with its smaller Gemma models to reduce memory consumption.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. http://arxiv.org/abs/2305.13245
* Introduces GQA, also from Google. Much more modern, from 2023. In the chapter we study it with Llama, the architecture that popularized it. 

Shazeer, N. (2020). GLU Variants Improve Transformer. http://arxiv.org/abs/2002.05202
* Comparing different GLUs. Introduces SwiGLU, the variant used by Llama, Gemma and most modern models.

Martra, P. (2024). EXPLORING GLU EXPANSION RATIOS: STRUCTURED PRUNING IN LLAMA-3.2 MODELS. https://doi.org/https://doi.org/10.31219/osf.io/qgxea
* Study the effect of pruning on the GLU structure of Llama-3.2 models.
