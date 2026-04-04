**The Challenge: The Retention Architect (HackNU 2026\)**

Participants are tasked with building an intelligent predictive system that not only identifies users at risk of leaving the platform but distinguishes between Voluntary Churn (user chooses to cancel) and Involuntary Churn (payment failures, technical issues).

***Task Requirements:***

Input: A synthetic/anonymized dataset containing user activity logs, payment history, generation activity, and demographic metadata.

Output: A Predictive Model: A classifier with a probability score for churn.

Insights: A list of possible reasons that explains why a specific user is likely to churn.

Strategy Proposal: A set of interventions based on model findings (e.g., specific discount triggers for voluntary churn vs. retry-logic optimization for involuntary churn).

***Additional Requirements:***

Explainability: An engine that justifies why the model flagged a user.

Categorization: The system must clearly separate "Passive/Involuntary" risks from "Active/Voluntary" risks.

Pipeline Flexibility: Teams can use any stack.

***Criteria:***  
\- Predictive Performance, 40%, "Accuracy, Precision-Recall AUC, and F1-score on the hidden test set."  
\- Actionable Insights, 40%, How well the team identified the root cause of churn and provided clear business recommendations.  
\- Presentation & UX, 20%, "Ability to translate complex data into a strategy a ""Product Manager"" could use."