# Rearchitecting LLMs. 
**SurgicalOptimization for Hyper-Efficient Models**

Are you using LLMs, or are you deciding how they should work? That's the difference between a user and an architect.
In this repository, you'll learn to go beyond using APIs or Frameworks; you'll learn to redesign the core of language models to make them faster, cheaper, and, above all, smarter for your specific use case.

You'll get to control what happens inside a model to the point where you'll be able to modify unwanted responses by changing just a few of the model's neurons.

This project is an advanced spin-off of the popular [Large Language Model Notebooks Course](https://github.com/peremartra/Large-Language-Model-Notebooks-Course) (+1800 stars), focused exclusively on optimization and re-architecture techniques that will allow you to go beyond traditional fine-tuning.

## The Journey: Your Path from User to LLM Architect
Being an LLM architect is a process. This repository is structured to guide you through the journey, from fundamentals to advanced techniques.

Step 1: Understanding the Rearchitecting Pipeline
* The Challenge: Understand the complete workflow to transform a generic model into a specialized one.
* Your Mission: Assimilate the three key phases of the process: Structural Optimization, Knowledge Recovery, and Specialization.
* Key Resource: (Under construction)

Step 2: Your First Surgery - Structural Pruning
* The Challenge: Perform your first structural modification on a model.
* Your Mission: Learn to identify and remove redundant components from an LLM (Depth Pruning) to make it lighter and faster, and then use Knowledge Distillation to "heal" the model and restore its performance.
* Key Resources: [CH02/CH02_NB01_Depth_pruning_evaluation.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB01_Depth_pruning_evaluation.ipynb) and [CH02/CH02_NB02_Knowledge_Recovery.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH02/CH02_NB02_Knowledge_Recovery.ipynb)

Step 3: From "Black Box" to Model Anatomy
* The Challenge: Stop seeing LLMs as a magic black box.
* Your Mission: Understand its internal structure, how information flows, and where its capabilities truly reside. You'll start thinking in terms of layers, blocks, modules, and neurons, not just prompts and responses.
* Key Resource: [CH03/CH03_NB01_Model_structures.ipynb](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH03/CH03_NB01_Model_structures.ipynb)

Step 4: Diving deeper into structural optimization.
* The Challenge: Adapt the model's structure to its mission and the data it will use.
* Your Mission: Implement depth and width pruning techniques on modern architectures.

Step 5: Creating a Specialist with LoRA.
* The Challenge: Create a domain-specialist model.
* Your Mission: Implement specialized fine-tuning with LoRA on SLMs.

Step 6: Controlling the Information Flow - Attention Bypass
* The Challenge: Go beyond static removal and start managing the information flow dynamically.
* Your Mission: Implement bypass mechanisms in the attention layers, allowing the model to selectively "ignore" calculations that are not necessary for certain tasks, achieving a drastic acceleration in inference.

Step 7: Mastery - Adaptive Attention & Fair Pruning
* The Challenge: Build systems that are not only efficient, but also aware and fair.
Your Mission:
* Adaptive Attention Bypass: Create systems that decide in real-time which attention layers to activate, adapting to the complexity of each input.
* Fair Pruning: Analyze the model's internal activations to identify and prune neurons that contribute to unwanted biases, building models that are efficient and ethical from their architecture.

## Your Architect's Toolkit
Inside this repository you'll find the concepts and code to master re-architecture techniques. To apply these techniques in practice and facilitate their use in production, all the methodologies are being consolidated into the [optiPfair](https://github.com/peremartra/optipfair) support library.

Tools you'll master:

* Structural Analysis: Visualization and understanding of architectures like LLaMA, Gemma, and Mistral.
* Structural Pruning: Depth Pruning (removing layers) and Width Pruning (reducing neurons).
* Knowledge Recovery: Knowledge Distillation to re-train optimized models.
* Advanced Attention Mechanisms: Implementation of bypass techniques for ultra-fast inference.
* Activation Analysis: The foundation for understanding internal behavior and applying Fair Pruning.

## Start Building
The journey begins with the first step. We recommend you start with the notebooks from Chapter 2 and 3 to build a solid foundation before tackling the more advanced techniques.

Stop being a mere user. It's time to become an architect.
