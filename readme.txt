

The data seem to display information about each customer, including customer ID, personal information, services, and billing.
Most of the features are yes/no answers, i.e. boolean


insights.append("FEATURE EXPLORATION: \n")
insights.append(f"-{churn_rate}% of costumers in the dataset have churned")
insights.append(f"-Each costumer pays ${mean_monthly_cost} per month on average\n")
insights.append(f"-The minimum monthly payment is ${min_monthly_payment}, which presumably is the price of the basis service\n")
insights.append(f"-The maximum monthly payment is ${max_monthly_payment}, which presumable is the price for all the services combined.
insights.append(f"-The average tenure is {mean_tenure} months for all costumers, and {mean_tenure_churn} months for costumers that have churned.\n")
                insights.append(f"-The average user subscribe to {mean_services} services\n")
                insights.append(f"-{round(churn_rate_by_contract[0]*100,2)}% of customers with a {churn_rate_by_contract.keys()[0]} contract have churned, ")
        insights.append(f"compared to {round(churn_rate_by_contract[1]*100,2)}% for customers with a {churn_rate_by_contract.keys()[1]}-contract ")
        insights.append(f"and {round(churn_rate_by_contract[2]*100,2)}% for customers with a {churn_rate_by_contract.keys()[2]}-contract")
#A majority of the costumers subscribe to the phone service (90%) and/or the internet service (78%) 
        #Electronic check have a higher churn rate than the other methods (45%, compared to 15-20% for automatic bank and credit card transfer, and mailed check)



Highpaying customers have higher churn rate than lowpaying (35% vs. 18%). 