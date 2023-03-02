'''
   first we ll go through the multiple python libraries 
   we use in data science 
'''

#1 - Matplotlib : data visualisation library, and most commonly used one 

#create a line chart with matplotlib

 #import the pyplot module from the matplotlib lib
from matplotlib import pyplot as plt 


#create a plot 
 #x axis and y axis 
plt.plot([1,2,3],[1,4,9])

#name the axis 
plt.xlabel('this is x axis')
plt.ylabel('this is y axis')
plt.legend(['Data Set 1','Data Set 2'])

plt.show()

#save the figure 
plt.savefig('myimage')

 #run the cell (shift + enter)



#2- Pandas : import , organize and process data 

from matplotlib impoty pyplot as plt 
import pandas as pd

 #create a dataframe 
data = {'year': [2008,2012,2016],
        'attendees': [112,321,729],
        'average age': [24,43,31]}

 #parse this data as a dataframe 

df = pd.DataFrame(data)

#select a single column 
df['year']
 
  #the result type -> pandas series 
type(df['year'])

#we can use it with boolean values 
 #example : test inequality 

earlier_than_2013 = df['year'] < 2013

#we can use something called boolean indexing 
df[earlier_than_2013] #shows only the rows that pass the condition 

#plot the year on the x axis , and the attendees on the y axis
plt.plot(df['year'], df['attendees'])
#more plots 
plt.plot(df['year'],df['average age'])
plt.legend(['attendees','average age'])
plt.show()



'''
Importing data with pandas

'''

import pandas as pd 
from matplotlib import pyplot as plt 

#import the csv file 
mydata = pd.read_csv('countries.csv')
#show the data 
data 
#or 
data.head() #when csv big

#access a column 
data['country'] #or 
data.country 

#boolean indexing (again)
algdata = data[data.country == 'Algeria']

#plot this data (using matplotlib)
plt.plot(algdata.year, algdata.pib)





'''
When to use histograms:

  Histograms help you understand the distribution of a 
  numeric value in a way that you cant with mean or median 
  alone 

'''

#create histograms in matplotlib 

import pandas as pd 
from matplotlib import pyplot as plt 

#import the csv
data = pd.read_csv('countries.csv')
data.head()


#set of unique values from a column 
set(data.continent) #africa,americas,asia,europe,oceania

data_2007 = data[data.year == 2007]

asia_2007 = data_2007[data_2007.continent == 'Asia']
europe_2007 = data_2007[data_2007.continent == 'Europe']

#check our data 
asia_2007.head()
europe_2007.head()


#check HOW MANY countries there are in these data sets
#use python set (useful)
print(len(set(asia_2007.country)))
print(len(set(europe_2007.country)))

#compare the distribution of pib of both continents
#use the mean and median (already implemented)
print('Mean pib in Asia')
print(asia_2007.pib.mean())

print('Mean pib in Europe')
print(europe_2007.pib.mean()) 

print('Median pib in Asia')
print(asia_2007.pib.median())




print('Median pib in Europe')
print(europe_2007.pib.median())


#create a histogram of pib in asia 
plt.hist(asia_2007.pib, 20, edgecolor = 'black')
plt.show()

#compare with europe pib 
plt.subplot(2,1,1) #2 rows + 1 column + select the first subplot 
#this can also be written : plt.subplot(211)

#add a title 
plt.title('Distribution of pib')

#hint: to compare two histograms set them to the same range
plt.hist(asia_2007.pib, 20, range=(0, 50000), edgecolor = 'black')
plt.ylabel('Asia')

plt.subplot(2,1,2)
plt.hist(europe_2007.pib,20, range=(0, 50000), edgecolor = 'black')
plt.ylabel('Europe')

plt.show()


'''

Practise histograms:

  Compare Europe and americas' life expectancy

'''

import pandas as pd 
from matplotlib import pyplot as plt 


 #read the csv data 
mydata = pd.read_csv('countries.csv')
mydata.head()

#organize data (get only europe and asia )
europe_data = mydata[mydata.continent == 'Europe']
americas_data = mydata[mydata.continent == 'Americas']

#check the data
europe_data.head()
americas_data.head()


#create histograms for both continents life exp

plt.subplot(2,1,1)

plt.title('Comparaison between life expectancy in Europe and Americas')

plt.ylabel('Europe')
plt.hist(europe_data.lifeExpectancy,20, range=(0,10000) , edgeColor='red')


plt.subplot(2,1,2)
plt.hist(americas_data.lifeExpectancy,20, range=(0,10000), edgeColor = 'blue' )
plt.ylabel('Americas')

plt.show()



'''
      Line Charts and Time Series

   Time Series : any chart that shows a trend over time 
   (usually a line chart)

   When to use them?
     - need to show a trend over time 
     - tests a hypothesis on a variety of conditions 
     - reduces misinterpretation in data 

  
     

'''

#compare gdp Per capita growth in the us and china 
  #-> comparison of a trend over time -> time series with a line chart

#get the right data 
import pandas as pd
from matplotlib import pyplot as plt

#get the csv
mydata2 = pd.read_csv('countries.csv')

#check our data 
mydata2.head()

#we only need usa and china's data

usa_data = mydata2[mydata2.country == 'United States']
usa_data.head()

#plot the usa gdpPerCapita with year on the x axis
plt.title('Usa gdpPerCapita Over the years')
plt.xlabel('year')
plt.ylabel('gdpPerCapita')
plt.plot(usa_data.year,usa_data.gdpPerCapita)
plt.show()



#grab china's data 
china_data = mydata2[mydata2.country == 'China']
china_data.head()


#plot china's gdpPerCapita With the years
plt.title('China gdpPerCapita Over the years')
plt.xlabel('year')
plt.ylabel('gdpPerCapita')
plt.plot(china_data.year,china_data.gdpPerCapita)
plt.show()


#we can display both plots 
plt.plot(usa_data.year,usa_data.gdpPerCapita)
plt.plot(china_data.year,china_data.gdpPerCapita)

#legend for both plots
plt.legend(['United States', 'China'])

plt.xlabel('year')
plt.ylabel('GDP per Capita')
plt.show()



#we want to see the growth in the gdpPerCapita comparing to the first year

#select an attribute with integer index in a pandas series
#gives us the first item in the usa_data.gdpPerCapita series
usa_data.gdpPerCapita.iloc[0] 

#get the % 
us_growth = usa_data.gdpPerCapita / usa_data.gdpPerCapita.iloc[0] * 100
china_growth = china_data.gdpPerCapita / china_data.gdpPerCapita.iloc[0] * 100

#compare between the growths -> plot 

plt.title('GDP Per Capita Growth (first year = 100)')

plt.plot(usa_data.year,us_growth)
plt.plot(china_data.year, china_growth)

plt.legend(['United States', 'China'])
plt.xlabel('year')
plt.ylabel('GDP per Capita growth')
plt.show()


'''

Another Practise Problem : 
  Compare Population Growth in the US and China 

'''

#import what we need 
import pandas as pd 
from matplotlib import pyplot as plt 

#get the data from the csv 
countriesData = pd.read_csv('countries.csv')
countriesData.head()

#get the data we need (us and china)
usa_data = countriesData[countriesData.country == 'United States']
china_data = countriesData[countriesData.country == 'China']

#plot the 2 populations over time 
plt.title('Usa and China populations over time')

plt.plot(usa_data.year,usa_data.population)

plt.xlabel('year')
plt.ylabel('Population')

plt.plot(china_data.year,china_data.population)

plt.legend(['United States','China'])

plt.show()


#compare the 2 population growth (with the first year )

us_population_growth = usa_data.population / usa_data.population.iloc[0] * 100
china_population_growth = china_data.population / china_data.population.iloc[0] * 100 

#plot the 2 growths
plt.title('Comparison between US and china Population Growth')

plt.plot(usa_data.year,us_population_growth)
plt.plot(china_data.year, china_population_growth)

plt.xlabel('year')
plt.ylabel('Population Growth')


plt.legend(['United States', 'China'])

plt.show()


'''
  Scatter plots :
   convenient way to visualize how two numeric variables are 
   related in your data.


   When to use scatter plots? 
      Examining relationships in data
    - helps you understand relationships between multiple 
    variables (eg: height and weight, work performance and experience ..etc)
 
    - helps you find outliers (eg: brillant workers with small experience
    but impressive performance)


  

'''


#create scatter plots with matplotlib 

#eg: find relationship between GDP per Capita and life expectancy 

#imports 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


#get the countries data  
mydata3 = pd.read_csv('countries.csv')
mydata3.head()


#lets examine how GDP per capita and life exp are related in 2007
data_2007 = mydata3[mydata3.year == 2007]
data_2007.head()

#create a scatter plot 
plt.title('Relationship between gdpPerCapita and life expectancy across countries in 2007')
plt.xlabel('GdpPerCapita')
plt.ylabel('LifeExp')
plt.scatter(data_2007.gdpPerCapita, data_2007.lifeExpectancy)
plt.show()


#find a correlation between them
data_2007.gdpPerCapita.corr(data_2007.lifeExpectancy)


#lets use log10 function to scatter a better plot 
#draw a plot with the log10 of gdpPerCapita
plt.scatter(np.log10(data_2007.gdpPerCapita), data_2007.lifeExpectancy)
plt.show()


#check the correlation 
np.log10(data_2007.gdpPerCapita).corr(data_2007.lifeExpectancy)


#run this analysis through every year in our data 

#find the set of all the years available
set(mydata3.year)

#sort them 
years_sorted = sorted(set(mydata3.year))

#loop
for given_year in years_sorted:
    plt.title(given_year)
    plt.scatter(np.log10(mydata3[mydata3.year == given_year].gdpPerCapita), mydata3[mydata3.year == given_year].lifeExpectancy)
    
    plt.xlim(2,5)
    plt.ylim(25,85)

    plt.xlabel('gdpPerCapita (log scale)')
    plt.ylabel('LifeExpec')
    #plt.show()
#lets export the plots as images , examining how they have changed over time
#first argument is a title (string)
    plt.savefig(str(given_year), dpi= 200) #dpi dots per inch
    plt.clf() #clear the current plot



'''
Practise problem:
  Examine the relationship between GDP and Life expectancy 
  in 2007 

'''

import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 


#get the data 
data = pd.read_csv('countries.csv')
data.head()

#get 2007 data 
data_2007 = data[data.year == 2007]
data_2007.head()

#scatter plot for gdp and life expec
plt.scatter(data_2007.gdpPerCapita*data_2007.population,data_2007.lifeExpectancy)
plt.title('Relationship between GDP and life expectancy')
plt.xlabel('GDP')
plt.ylabel('Life expectancy')
plt.show()


'''
  Bar Charts : 
    - convenient way to compare numeric values of SEVERAL GROUPS 
    - Good for comparing multiple values 
    - gives u new insight

'''


#compare the populations of the 10 most populous countries in 2007 

import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np 

data = pd.read_csv('countries.csv')
data.head()

data_2007 = data[data.year == 2007]
data_2007.head()

#sort the dataframe according to population 
data_2007.sort_values('population', ascending=False)

#select the 10 most populous countries 
mostPopCountries2007 = data_2007.sort_values('population', ascending=False).head(10)


#plot the bar
 #set the range 
x = range(10) #same as x = [1,2,3.....]
plt.title('Populations of 10 most populous countries in 2007')
plt.bar(x, mostPopCountries2007.population / 10**6) #divide by a million to make the graph easier to read
#set the ticks (on the bar chart)
plt.xticks(x, mostPopCountries2007.country, rotation='vertical')
#set the labels 
plt.xlabel('Country')
plt.ylabel('Population in millions')
plt.show()



'''
 Practise problem: 
   Compare the GDP of the 10 Most populous countries in 2007

'''

import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np 


#get the data 
data = pd.read_csv('countries.csv')
data.head()

data_2007 = data[data.year == 2007]
data_2007.head()


#get the top 10 of most populous countries 
top10 = data_2007.sort_values('population', ascending=False).head(10)
top10.head()


#plot the bar chart 
#set the range 
x= range(10)
plt.title('GDP of the 10 most populous countries in 2007')

plt.bar(x, top10.gdpPerCapita * top10.population)

#set the ticks and labels 
plt.xticks(x,top10.country, rotation='vertical')
plt.xlabel('Countries')
plt.ylabel('GDP')
plt.show()



#lets add another bar chart to this plot 
plt.subplot(2,1,1) #graph = 2 rows and 1 column of subplots

x= range(10)

plt.bar(x,top10.population / 10**6)


#for multiple ticks 
plt.xticks([],[]) 

plt.title('Comparison between population and gdp of top 10 mpc's in 2007')

plt.legend(['Population in millions'])

#select the second plot
plt.subplot(2,1,2)

plt.bar(x, top10.gdpPerCapita * top10.population / 10**9)

#set the ticks and labels 
plt.xticks(x,top10.country, rotation='vertical')

plt.legend(['GDP in Billions'])

plt.xlabel('Countries')
plt.ylabel('GDP')
plt.show()
















