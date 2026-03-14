**Chapter 7: Efficient Fine-Tuning for SLMs (Work in Progress)**
This chapter will cover the specialization stage of the rearchitecting pipeline — the final step that transforms a pruned and distilled model into a domain-ready solution.
The chapter will follow a natural progression from first principles to production-ready techniques:

* Low-Rank Decomposition: Building the intuition behind why we can approximate weight updates with a fraction of the parameters.
* LoRA: The foundational technique for efficient fine-tuning, implemented hands-on.
* DoRA: Weight-decomposed LoRA and how it improves upon the original approach.
* Quantization-aware fine-tuning: Combining precision reduction with adaptation.
* QDoRA: The production-ready combination that closes the rearchitecting pipeline.
