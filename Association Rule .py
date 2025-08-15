{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "10c39b74-46b9-4345-b3ec-560eb2167fbf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from mlxtend.frequent_patterns import apriori,association_rules\n",
    "from mlxtend.preprocessing import TransactionEncoder\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2c1ecc90-b719-4673-930f-eaa8b200295d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('Online retail.csv',header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ef7ad72f-42db-42b6-9727-d7a6cc044d7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7501, 1)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f4d26c05-ba13-4fa0-87af-a352b8f2e2af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>shrimp,almonds,avocado,vegetables mix,green gr...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>burgers,meatballs,eggs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>chutney</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>turkey,avocado</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>mineral water,milk,energy bar,whole wheat rice...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   0\n",
       "0  shrimp,almonds,avocado,vegetables mix,green gr...\n",
       "1                             burgers,meatballs,eggs\n",
       "2                                            chutney\n",
       "3                                     turkey,avocado\n",
       "4  mineral water,milk,energy bar,whole wheat rice..."
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7c4b6c8e-c30d-4e2a-a485-96d42a7d4f37",
   "metadata": {},
   "outputs": [],
   "source": [
    "transmation = df[0].apply(lambda x: x.split(','))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2e446eee-9f16-40a7-be96-76150f17248b",
   "metadata": {},
   "outputs": [],
   "source": [
    "trans_list = transmation.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "15fc7280-ebfc-4575-a023-9500ec7f7bb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "te = TransactionEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1f874f6d-f87b-4b58-8a96-71d23f460469",
   "metadata": {},
   "outputs": [],
   "source": [
    "trans_data=te.fit(trans_list).fit_transform(trans_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2c09651e-94a6-4fa2-8372-dd3edb366e9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.DataFrame(trans_data,columns=te.columns_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2013eebc-be66-4422-9d70-969b2f88a57f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2347"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bb7b4cdc-804c-4ca5-8c00-01a86ee573d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop_duplicates(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4653e9a1-eba5-4d4a-a9af-049dc859e79f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "46c69ece-6841-4ee2-97f7-57de47d058e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "frequent = apriori(data,min_support=0.03,use_colnames=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "595fab1f-d2ed-47cf-9efc-8e4b434c7622",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>support</th>\n",
       "      <th>itemsets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.046178</td>\n",
       "      <td>(avocado)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.045208</td>\n",
       "      <td>(brownies)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.114280</td>\n",
       "      <td>(burgers)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.041327</td>\n",
       "      <td>(butter)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.103609</td>\n",
       "      <td>(cake)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>89</th>\n",
       "      <td>0.034730</td>\n",
       "      <td>(tomatoes, mineral water)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>90</th>\n",
       "      <td>0.032596</td>\n",
       "      <td>(spaghetti, olive oil)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>91</th>\n",
       "      <td>0.036282</td>\n",
       "      <td>(spaghetti, pancakes)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>92</th>\n",
       "      <td>0.030462</td>\n",
       "      <td>(spaghetti, shrimp)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>93</th>\n",
       "      <td>0.030074</td>\n",
       "      <td>(spaghetti, tomatoes)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>94 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     support                   itemsets\n",
       "0   0.046178                  (avocado)\n",
       "1   0.045208                 (brownies)\n",
       "2   0.114280                  (burgers)\n",
       "3   0.041327                   (butter)\n",
       "4   0.103609                     (cake)\n",
       "..       ...                        ...\n",
       "89  0.034730  (tomatoes, mineral water)\n",
       "90  0.032596     (spaghetti, olive oil)\n",
       "91  0.036282      (spaghetti, pancakes)\n",
       "92  0.030462        (spaghetti, shrimp)\n",
       "93  0.030074      (spaghetti, tomatoes)\n",
       "\n",
       "[94 rows x 2 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "frequent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "913c9152-8d13-4273-84c3-df6993ee6234",
   "metadata": {},
   "outputs": [],
   "source": [
    "rule = association_rules(frequent,metric='lift',min_threshold=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "30000d88-03ab-441b-b49f-e44f73ac12d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>antecedents</th>\n",
       "      <th>consequents</th>\n",
       "      <th>antecedent support</th>\n",
       "      <th>consequent support</th>\n",
       "      <th>support</th>\n",
       "      <th>confidence</th>\n",
       "      <th>lift</th>\n",
       "      <th>representativity</th>\n",
       "      <th>leverage</th>\n",
       "      <th>conviction</th>\n",
       "      <th>zhangs_metric</th>\n",
       "      <th>jaccard</th>\n",
       "      <th>certainty</th>\n",
       "      <th>kulczynski</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>(eggs)</td>\n",
       "      <td>(burgers)</td>\n",
       "      <td>0.207994</td>\n",
       "      <td>0.114280</td>\n",
       "      <td>0.036282</td>\n",
       "      <td>0.174440</td>\n",
       "      <td>1.526427</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.012513</td>\n",
       "      <td>1.072872</td>\n",
       "      <td>0.435445</td>\n",
       "      <td>0.126866</td>\n",
       "      <td>0.067922</td>\n",
       "      <td>0.245964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>(burgers)</td>\n",
       "      <td>(eggs)</td>\n",
       "      <td>0.114280</td>\n",
       "      <td>0.207994</td>\n",
       "      <td>0.036282</td>\n",
       "      <td>0.317487</td>\n",
       "      <td>1.526427</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.012513</td>\n",
       "      <td>1.160427</td>\n",
       "      <td>0.389373</td>\n",
       "      <td>0.126866</td>\n",
       "      <td>0.138248</td>\n",
       "      <td>0.245964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>(burgers)</td>\n",
       "      <td>(mineral water)</td>\n",
       "      <td>0.114280</td>\n",
       "      <td>0.299961</td>\n",
       "      <td>0.034730</td>\n",
       "      <td>0.303905</td>\n",
       "      <td>1.013147</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.000451</td>\n",
       "      <td>1.005666</td>\n",
       "      <td>0.014651</td>\n",
       "      <td>0.091513</td>\n",
       "      <td>0.005634</td>\n",
       "      <td>0.209844</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>(mineral water)</td>\n",
       "      <td>(burgers)</td>\n",
       "      <td>0.299961</td>\n",
       "      <td>0.114280</td>\n",
       "      <td>0.034730</td>\n",
       "      <td>0.115783</td>\n",
       "      <td>1.013147</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.000451</td>\n",
       "      <td>1.001699</td>\n",
       "      <td>0.018537</td>\n",
       "      <td>0.091513</td>\n",
       "      <td>0.001696</td>\n",
       "      <td>0.209844</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>(spaghetti)</td>\n",
       "      <td>(burgers)</td>\n",
       "      <td>0.230113</td>\n",
       "      <td>0.114280</td>\n",
       "      <td>0.030462</td>\n",
       "      <td>0.132378</td>\n",
       "      <td>1.158361</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.004164</td>\n",
       "      <td>1.020859</td>\n",
       "      <td>0.177573</td>\n",
       "      <td>0.097033</td>\n",
       "      <td>0.020433</td>\n",
       "      <td>0.199466</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       antecedents      consequents  antecedent support  consequent support  \\\n",
       "0           (eggs)        (burgers)            0.207994            0.114280   \n",
       "1        (burgers)           (eggs)            0.114280            0.207994   \n",
       "2        (burgers)  (mineral water)            0.114280            0.299961   \n",
       "3  (mineral water)        (burgers)            0.299961            0.114280   \n",
       "4      (spaghetti)        (burgers)            0.230113            0.114280   \n",
       "\n",
       "    support  confidence      lift  representativity  leverage  conviction  \\\n",
       "0  0.036282    0.174440  1.526427               1.0  0.012513    1.072872   \n",
       "1  0.036282    0.317487  1.526427               1.0  0.012513    1.160427   \n",
       "2  0.034730    0.303905  1.013147               1.0  0.000451    1.005666   \n",
       "3  0.034730    0.115783  1.013147               1.0  0.000451    1.001699   \n",
       "4  0.030462    0.132378  1.158361               1.0  0.004164    1.020859   \n",
       "\n",
       "   zhangs_metric   jaccard  certainty  kulczynski  \n",
       "0       0.435445  0.126866   0.067922    0.245964  \n",
       "1       0.389373  0.126866   0.138248    0.245964  \n",
       "2       0.014651  0.091513   0.005634    0.209844  \n",
       "3       0.018537  0.091513   0.001696    0.209844  \n",
       "4       0.177573  0.097033   0.020433    0.199466  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rule.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14fc73d7-ced4-4f3c-820f-1bb7fb122031",
   "metadata": {},
   "source": [
    "* lift tells how strong the relationship b/w items in a tranaction\n",
    "* for example if a person buy a bread how they likely to also buy butter\n",
    "* Support\n",
    "* tells how frequently items appears in a dataset\n",
    "* ex: if 30 people buy a milk out of 100 trans support is 30/100 = 0.3\n",
    "* confidence:\n",
    "* tells how often y is buy when x is buy\n",
    "* challages in association rule\n",
    "* too many rules\n",
    "* slow for big data\n",
    "* maybe difficult to choose correct thresholds\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61fc86da-0c7f-4416-8d7b-3bf806d05b48",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
