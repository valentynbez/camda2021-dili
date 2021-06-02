# camda2021-dili
![CAMDA2021-logo](http://camda2021.bioinf.jku.at/lib/tpl/mnml-blog/user/logo.png)
<br>
**CAMDA 2021 solution for "Literature AI for Drug Induced Liver Injury" challenge**

*All data for the project can be found on an official [CAMDA website](http://camda2021.bioinf.jku.at/doku.php).*

**Description:**

"Unexpected Drug-Induced Liver Injury (DILI) still is one of the main killers of promising novel drug candidates. It is a clinically significant disease that can lead to severe outcomes such as acute liver failure and even death. It remains one of the primary liabilities in drug development and regulatory clearance due to the limited performance of mandated preclinical models even today. The free text of scientific publications is still the main medium carrying DILI results from clinical practice or experimental studies. The textual data still has to be analysed manually. This process, however, is tedious and prone to human mistakes or omissions, as results are very rarely available in a standardized form or organized form. There is thus great hope that modern techniques from machine learning or natural language processing could provide powerful tools to better process and derive the underlying knowledge within free form texts. The pressing need to faster process potential drug candidates in the current COVID epidemic combined with recent advances in Artificial Intelligence for text processing make this Challenge particularly topical.

We have compiled a large set of PubMed papers relevant to DILI (positives) to be contrasted with a challenging set of unrelated papers (negatives). Both titles and abstracts have been collected. Can you build a classifier using modern AI or NLP techniques to identify the relevant papers?

The positive reference data set comprises of ~14,000 DILI related papers referenced in the NIH LiverTox database, which have been validated by a panel of DILI experts. This positive reference is split 50:50 into one part released for the challenge and one part withheld part for final performance testing.
This is complemented by a realistic, non-trivial negative reference set of ~14,000 papers that is highly enriched in manuscripts that are not relevant to DILI but where obvious negatives and any positives we could identify have been removed by filtering for keywords and through well established language models, followed by a selective manual review by DILI experts at the FDA. This negative reference is also split 50:50 into one part released for the challenge and one part withheld part for final performance testing.
Together, this thus recreates the problem faced by human experts: After the obvious, easy negatives and positives have been removed by basic algorithms, how can we identify true positives and negatives for the less obvious cases?

The released data should be used for both training and (nested) cross-validation to avoid over-fitting. Participants will then receive independent performance scores from the withheld additional test data.

Considering that the overall prevalence of DILI relevant papers is very low when considering all manuscripts in PubMed, we will also provide another independent performance score where the negative reference set has been expanded considerably to provide an assessment of how well the models can be applied to larger candidate collections that are naturally highly unbalanced."
