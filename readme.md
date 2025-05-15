# KG-QAGen: A Knowledge-Graph-Based Framework for Systematic Question Generation and Long-Context LLM Evaluation

<p align="center">
  <a href="https://example.com/your-homepage">
    <img src="https://img.shields.io/badge/Homepage-Link-blue" alt="🌐 Homepage" />
  </a>
  <a href="https://huggingface.co/datasets/gtfintechlab/KG-QAGen-D">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="🤗 Dataset" />
  </a>
  <a href="https://arxiv.org/abs/YYYY.MM.NNNNN">
    <img src="https://img.shields.io/badge/arXiv-YYYY.MM.NNNNN-red?logo=arxiv" alt="📖 arXiv" />
  </a>
  <a href="https://github.com/gtfintechlab/KG-QAGen">
    <img src="https://img.shields.io/badge/GitHub-KG--QAGen-black?logo=github" alt="🐙 GitHub" />
  </a>
</p>

---

KG‑QAGen is a benchmark and toolkit that leverages structured annotations of financial agreements to build knowledge graphs and automatically generate QA pairs at controlled difficulty levels, enabling fine‑grained evaluation of long‑context LLMs.

<p align="center">
  <img src="figures/figure_1.png" width="100%" alt="KG‑QAGen Overview" />
</p>
--
Overview of KG-QAGEN. Credit agreements are annotated to identify entities and their
relationships, forming a knowledge graph representation. This graph is then used to systematically
extract multi-level QA pairs, which serve as the basis for benchmarking long-context LLMs.
---

## KG‑QAGen‑D Dataset

We release **KG‑QAGen‑D**, a 16,116-question benchmark derived from 170 SEC credit agreements (2013–2022). Each QA pair is tagged with a composite complexity level (L = #hops + #set‑ops + plurality), split into *Easy*, *Medium*, and *Hard*.

---

## Leaderboard & Evaluation Platform

To facilitate reproducibility and future research, we release the **KG‑QAGen‑D** dataset under a [CC-BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/). The dataset is divided into development and test sets as follows:

| **Stats**                 |   **Dev** |   **Test** |  **Total** |
| ------------------------- | --------: | ---------: | ---------: |
| # Documents               |        40 |        130 |        170 |
| # Questions per Doc (Min) |         5 |          2 |          2 |
| # Questions per Doc (Avg) |     39.71 |      33.19 |      34.54 |
| # Questions per Doc (Max) |       230 |        153 |        230 |
| # Easy Questions          |     1,289 |      4,917 |      6,206 |
| # Medium Questions        |     1,583 |      7,958 |      9,541 |
| # Hard Questions          |       143 |        226 |        369 |
| **Total Questions**       | **3,015** | **13,101** | **16,116** |

* **Development Set (20%)**: 40 documents and 3,015 QA pairs are publicly released to support model development and validation.
* **Test Set (80%)**: 130 documents and 13,101 QA pairs are **not released** to prevent data contamination and ensure fair evaluation.

### Online Leaderboard

We host an evaluation leaderboard on **[Hugging Face](https://huggingface.co/spaces/gtfintechlab/KG-QAGen-Leaderboard)**. To participate:

1. Generate predictions on the test set using the provided document context.

2. Format your results as a JSON file:

   ```json
   [
     {
       "question_id": "kgqa_test_0001",
       "output": "Your model's answer here"
     },
     ...
   ]
   ```

3. Submit your JSON file via the Hugging Face interface.

4. The platform will automatically evaluate your predictions against hidden ground truth.


## Contact

For questions or issues, please reach out to:

- Nikita Tatarinov: [ntatarinov3@gatech.edu](mailto:ntatarinov3@gatech.edu)
- Agam Shah: [ashah482@gatech.edu](mailto:ashah482@gatech.edu)

---

## Citation

If you use KG‑QAGen in your work, please cite:

```bibtex
@inproceedings{tatarinov2025kgqagen,
  title     = {{KG‑QAGen}: A Knowledge‑Graph‑Based Framework for Systematic Question Generation and Long‑Context LLM Evaluation},
  author    = {Tatarinov, Nikita and Kannan, Vidhyakshaya and Srinivasa, Haricharana and Raj, Arnav and Anand, Harpreet and Singh, Varun and Luthra, Aditya and Lade, Ravij and Shah, Agam and Chava, Sudheer},
  booktitle = {NeurIPS Dataset and Benchmark},
  year      = {2025},
  url       = {https://github.com/gtfintechlab/KG-QAGen}
}
