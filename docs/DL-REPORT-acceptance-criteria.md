# Deep Learning Report Questions and Acceptance Criteria

## 1. Introduction

### Questions:
- How does your Deep Learning (DL) portfolio fit into the minor program?  
- Why is DL important in your areas of interest?  
- How does this DL project relate to your main project in the minor?  
- What are your learning objectives for this project?  

### Acceptance Criteria:
- DL relation to the minor is discussed.
- DL portfolio relation to main project in the minor is discussed.
- Size: Max 1 A4.

---

## 2. Problem Statement

### Questions:
- What is the updated **specific and measurable objective** of your project (revised from your ML portfolio)?  
- List and prioritize the **functional and technical requirements** for your project.  
- How will you test or validate whether your final result meets each requirement?  

### Acceptance Criteria:
- Problem definition is specific and measurable.
- Functional and technical requirements are listed and prioritized.
- Size: Max 1 A4.

---

## 3. Data Augmentation and Preprocessing

### Questions:
- Which **data augmentation methods** did you apply to your dataset? Why did you choose these methods?  
- Describe your **preprocessing pipeline** (e.g., image resizing, normalization, color adjustments).  
  - How does each preprocessing step improve clarity or reduce complexity for CNNs?  
- Provide **before-and-after examples** of images processed through your pipeline.  

### Acceptance Criteria:
- Data augmentation methods are used and explained.
- Preprocessing pipeline is implemented.
- Each preprocessing step is explained and justified.
- Size: Max 3 A4.

---

## 4. CNN Architecture, Training, and Validation

### Questions:
- **Design choices**:  
  - Did you design a custom CNN or use transfer learning?  
  - For a custom CNN:  
    - How many layers did you use?  
    - What activation functions and pooling methods did you select?  
  - For transfer learning:  
    - Which pretrained model did you use?  
    - Which layers were frozen, and why?  
    - How were new top layers designed for your task?  
- How did you prevent overfitting (e.g., dropout, early stopping)?  
- What **performance metrics** (accuracy, F1-score, etc.) did you use? Why?  
- Show a **confusion matrix** and discuss trade-offs (e.g., precision vs. recall).  
- Visualize **feature maps** from at least two layers and explain what they detect.  

### Acceptance Criteria:
- Architecture is designed and argued.
- Data is split into stratified subsets and checked.
- CNN is trained, cross-validated, and fine-tuned.
- Performance is evaluated using appropriate methods.
- Visualization of network's internal representations is provided.
- Size: Max 5 A4.

---

## 5. Deployment and Testing

### Questions:
- **Deployment setup**:  
  - Where is your model deployed (e.g., Raspberry Pi, cloud)?  
  - How does your preprocessing and prediction pipeline work in deployment?  
- **Test plan**:  
  - What requirements are you measuring (e.g., accuracy, inference speed)?  
  - What are your target performance levels?  
- **Test results**:  
  - How do your results compare to the targets?  
  - Were there unexpected behaviors or limitations?  

### Acceptance Criteria:
- Preprocessing and prediction pipeline deployed.
- Test plan present.
- Documentation of test results.
- Size: Max 5 A4.

---

## 6. Conclusion

### Questions:
- Summarize the main steps and key decisions in your project.  
- Did your results meet your SMART objectives? If not, why?  
- How well does your model generalize to unseen data (training vs. test performance)?  
- What worked well, and what would you improve in future iterations?  

### Acceptance Criteria:
- Results are compared to initial goals and SMART objectives.
- Generalization performance is analyzed.
- Size: Max 1 A4.

---

## 7. References

### Questions:
- List all references (e.g., papers, tutorials, tools) used in your project.  

### Acceptance Criteria:
- References to the sources used are provided.

---

## 8. Code Appendices

### Questions:
- Provide **code snippets** for:  
  - Data preprocessing  
  - Model architecture  
  - Training loop  
  - Evaluation methods  
- Explain the purpose of each snippet and justify your coding choices.  

### Acceptance Criteria:
- Code snippets are provided for key parts of the project.
- Code quality is sufficient.
