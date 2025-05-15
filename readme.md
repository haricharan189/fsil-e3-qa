# KG‑QAGen  
**A Knowledge‑Graph‑Based Framework for Systematic Question Generation and Long‑Context LLM Evaluation**

[![🌐 Homepage](https://img.shields.io/badge/Homepage-website-blue)](https://example.com/your-homepage)  
[![🤗 Dataset](https://img.shields.io/badge/HuggingFace-dataset-yellow)](https://huggingface.co/datasets/gtfintechlab/KG-QAGen-D)  
[![📖 arXiv](https://img.shields.io/badge/arXiv-YYYY.MM.NNNNN-BB0000?logo=arxiv)](https://arxiv.org/abs/YYYY.MM.NNNNN)  
[![GitHub](https://img.shields.io/badge/GitHub-KG%2DQAGen-181717?logo=github)](https://github.com/gtfintechlab/KG-QAGen)

KG‑QAGen is a benchmark and toolkit that leverages structured annotations of financial agreements to build knowledge graphs and automatically generate QA pairs at controlled difficulty levels, enabling fine‑grained evaluation of long‑context LLMs.

<p align="center">
  <img src="figures/overview.png" width="100%" alt="KG‑QAGen overview" />
</p>

---

## 📂 KG‑QAGen‑D Dataset

We release **KG‑QAGen‑D**, a 16 116‑question benchmark derived from 170 SEC credit agreements (2013–2022). Each QA pair is tagged with a composite complexity level (L = #hops + #set‑ops + plurality), split into *Easy*, *Medium*, and *Hard*.
