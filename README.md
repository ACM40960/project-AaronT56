# Exploring Predictive Insights of Philosophical Sentiment Analysis

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-available-yellow)

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Challenges](#challenges)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Support](#support)
- [Maintainers](#maintainers)
- [Acknowledgments](#acknowledgments)

## Project Overview

The intersection of sentiment analysis and philosophy presents a unique challenge due to the abstract and nuanced nature of philosophical texts. Traditional sentiment analysis models are typically designed to assess the emotional tone of simpler textual media, but their effectiveness in classifying texts that convey complex philosophical ideas remains unclear. This project explores whether existing sentiment analysis models can meaningfully classify philosophical texts and, if not, seeks to develop a specialized classification model tailored for this purpose.

## Objectives

1. **Data Acquisition**: Collect philosophical texts, determine the general philosophy underlying these texts, and convert them into text files suitable for analysis.
2. **Sentiment Analysis**: Evaluate the utility of various sentiment analysis models in distinguishing different philosophical beliefs within the texts.
3. **Model Development**: Develop a new classification model, specifically fine-tuned on the philosophical data, to improve classification accuracy.

## Methodology

1. **Sentiment Analysis Models**: The following sentiment analysis models were applied to the philosophical texts:
    - **TextBlob**
    - **VADER**
    - **Flair**
    - **Afinn**
    - **SenticNet**
    - **Transformer-based model**
    - **Pattern**
  
2. **Model Performance**: Each model was evaluated on its ability to predict three philosophical themes: Nihilism, Romanticism, and Stoicism. An accuracy metric was created such that the nihilism, romanticism and stoicism scores could be compared to outputs from the BERT model which was a binary classifier.

3. **Accuracy Metric Meaning**: In order to obtain some meaningful accuracies such that this model can be compared to the BERT binary classification model which was fine tuned, some accuracy metric was required. To obtain this accuracy, the number of 'correctly' classified points was had to be defined. To do this, each philosophical aspect was described as 'positive' or 'negative'. For example, nihilism was considered negative as you may expect the types of narratives in nihilistic texts to have an overall lower sentiment than non-nihilistic texts/philosophers. So, in the nihilism graph, find the lowest sentiment score texts, and check what portion of them have been categorised as nihilistic. For the non-nihilistic texts do the reverse. Using both of these scores, you can obtain the overall number of accuractely classified texts and hence, the overall accuracy. The same process is applied to romanticism and stoicism, where a romantic text is expected to have a higher sentiment score and a stoic text is expected to have lower sentiment scores compared to non-stoic. This permits accuracy measurements to be obtained for each model.

4. **Fine-Tuning BERT**: To address data limitations and enhance model performance, I generalized the three philosophical themes into a binary classification (positive vs. negative). A fine-tuning layer was applied to BERT using this binary classification approach, which permitted the generation of an accuracy score. The data was split with a 70-15-15 ratio, retaining sufficient data for validation and testing.
## Getting Started

To get started with this project:

1. **Clone the repository**:
    ```bash
    https://github.com/ACM40960/project-AaronT56.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the sentiment analysis**:
    Open the Jupyter Notebook `Sentiment_Analysis_Model_Testing.ipynb` and run all cells. This notebook contains the code to test different sentiment analysis models on the philosophical texts. Note: Obtaining the sentiment analysis scores will take large amounts of time (approximately 30-60 minutes each).

4. **Fine-tune the BERT model**:
    Open the Jupyter Notebook `BERT_Fine_Tuned_Model.ipynb` and run all cells. This notebook contains the code for fine-tuning the BERT model using the provided data.

## Challenges

- **Data Acquisition**: Finding and preparing philosophical texts for analysis.
- **Categorization**: Accurately categorizing texts by their underlying philosophies.
- **Complex Language**: Addressing the sophisticated language and abstract concepts present in philosophical texts.
- **Generational Differences**: Managing variations in authorial tone across different time periods.

## Results

- **Flair Model Performance**: Among the sentiment analysis models, Flair performed the best with accuracies of 0.67 for Nihilism, 0.73 for Romanticism, and 0.60 for Stoicism. Below are the visual representations of Flair's performance across the three philosophical themes:

### Nihilism
![Flair Sentiment Analysis for Nihilism](https://github.com/ACM40960/project-AaronT56/blob/main/plots_nihilism/mean_sentiment_score_flair_nihilism_plot.png)

### Romanticism
![Flair Sentiment Analysis for Romanticism](https://github.com/ACM40960/project-AaronT56/blob/main/plots_romanticism/mean_sentiment_score_flair_romanticism_plot.png)

### Stoicism
![Flair Sentiment Analysis for Stoicism](https://github.com/ACM40960/project-AaronT56/blob/main/plots_stoicism/mean_sentiment_score_flair_stoicism_plot.png)

Here are the obtained accuracy values for the rest of the models:

| Model       | Nihilism | Romanticism | Stoicism |
|-------------|----------|-------------|----------|
| TextBlob    | 0.58     | 0.55        | 0.30     |
| VADER       | 0.50     | 0.36        | 0.40     |
| Flair       | 0.67     | 0.73        | 0.60     |
| Afinn       | 0.42     | 0.36        | 0.40     |
| SenticNet   | 0.58     | 0.27        | 0.40     |
| Transformer | 0.58     | 0.64        | 0.50     |
| Pattern     | 0.58     | 0.55        | 0.30     |

- **BERT Fine-Tuning**: The BERT model, fine-tuned for binary classification, achieved an accuracy of 64%. This result suggests that sentiment analysis can be effectively applied to philosophical texts when tailored to the specific challenges of abstract and nuanced content with some degree of accuracy. As the data was balanced, this result is meaningful as a meaningless model would produce an accuracy of 50% by just arbitrarily guessing.

## Conclusion

This project developed a fine-tuned BERT classification model that achieved a 64% accuracy, outperforming many existing sentiment analysis tools in predicting the beliefs of various philosophers. The Flair model also demonstrated strong performance, with accuracies of 0.67, 0.73, and 0.60 for Nihilism, Romanticism, and Stoicism, respectively. Although data scarcity posed challenges, techniques such as K-fold cross-validation and regularization helped generate meaningful results. This project highlights the potential for applying sentiment analysis to complex philosophical texts, serving as a first step toward linking sentiment with underlying beliefs or mental states.

## Future Work

- **Expand Data Collection**: Gather more texts across additional philosophical themes to improve model diversity and accuracy.
- **Refine Models**: Further develop and fine-tune models to better handle the unique challenges posed by philosophical texts.
- **Explore Other Media**: Apply techniques to less complex media like books or TV shows to evaluate the consistency of the findings and perhaps achieve higher accuracies.
- **Incremental Improvements**: Experiment with small data augmentations or preprocessing techniques to enhance model performance on existing tasks. Encourage small, focused contributions to incrementally refine the model and dataset.
- **Sentiment Analysis and Mental State**: This project has created an understanding that sentiment can be attached to the underlying philosophies a given text has spoken about, and hence the philosophical beliefs a person might hold. Perhaps, if one could explore further ways in which a person's psychological state could be determined solely through the way they speak. If data were available, perhaps with a well-trained model, one could infer the psychological or philosophical prespectives of a person entirely through monitoring their word choices.
## Project Structure

- `Sentiment_Analysis_Model_Testing.ipynb`: Jupyter Notebook that contains the implementation of various sentiment analysis models, including TextBlob, VADER, Flair, Afinn, SenticNet, Transformer-based model, and Pattern. This notebook is used to test and compare the performance of these models on different philosophical texts.

- `BERT_Fine_Tuned_Model.ipynb`: Jupyter Notebook that implements the fine-tuning of the BERT model for binary classification on the philosophical text data. This notebook generalizes philosophical themes into a binary classification (positive vs. negative) and trains the model accordingly.

- `Abbreviations.ipynb`: Jupyter Notebook used to create the .py file that will be used to create the naming abbreviations of the texts for the x-axis of the graph.

- `belief_maps.ipynb`: This notebook creates the .py file which is used to map each of the texts in the data set to the belief which they have been categorised into. It is simply a long dictionary.

- `abbreviations.py`: Python script that contains functions and data for handling abbreviations related to the philosophical texts (Nihilism, Romanticism, Stoicism).

- `belief_maps.py`: This is the file which is used to map the underlying philosophical beliefs of a text with the texts in the data set.

- `All_Texts/`: Directory containing all of the collected philosophical texts used in the analysis, organized by philosophical theme (Nihilism, Romanticism, Stoicism).

- `plots_nihilism/`: Directory containing plots generated from the analysis of Nihilism texts, including sentiment analysis scores.

- `plots_romanticism/`: Directory containing plots generated from the analysis of Romanticism texts, including sentiment analysis scores.

- `plots_stoicism/`: Directory containing plots generated from the analysis of Stoicism texts, including sentiment analysis scores.

- `Nihilism/`: Directory containing texts specifically related to Nihilism that were analyzed in the project.

- `Romanticism/`: Directory containing texts specifically related to Romanticism that were analyzed in the project.

- `Stoicism/`: Directory containing texts specifically related to Stoicism that were analyzed in the project.

- `nihilism_sentiment_analysis.csv`: CSV file containing sentiment analysis results for the Nihilism texts.

- `romanticism_sentiment_analysis.csv`: CSV file containing sentiment analysis results for the Romanticism texts.

- `stoicism_sentiment_analysis.csv`: CSV file containing sentiment analysis results for the Stoicism texts.

- `requirements.txt`: File containing a list of Python packages and dependencies required to run the project.

- `README.md`: The current README file providing an overview of the project, setup instructions, and a detailed description of the project structure.

## Support

If you have any questions or need help, feel free to open an issue in the repository or contact the maintainer (myself) at aaron.timony@ucdconnect.com.

## Maintainers

This project is maintained by [Aaron Timony](https://github.com/AaronT56).

## Acknowledgments

I would like to express my gratitude to everyone who supported this project:

- **University College Dublin**: For providing the opportunity to undertake this project as part of my master’s program.
- **Sarp Akcay**: My module supervisor, for his guidance and weekly lectures that provided valuable insights into the methodologies and tools necessary to complete this project.
- **Open-source Community**: For developing and maintaining the libraries and tools that made this project possible, including `nltk`, `torch`, `transformers`, `flair`, `scikit-learn`, `matplotlib`, `seaborn`, and others.
  
Lastly, I would like to acknowledge the broader research and development community whose work in sentiment analysis, natural language processing, and machine learning laid the foundation for this project.


