# Laboration Maskininlärning

## EDA - in exploratory_data_analysis.py

#### 1.1 a)
Jag gjorde började med en initiell analys mha info(), describe(), head(), tail(), df.columns, df.index. min(), max(), unique för varje label. 
Därefter hämtade jag sedan ut de unika kolumnnamnen då de var många i respektive fil, skriver inte ut dom utan kollar via debugg för att få översyn.
Gjorde därefter statistic på totalen med np.stats.describe
Kollar min och max värden på relevanta kolumner mha funktion, det gör jag regelbundet under utvecklingen men alla anrop till funktionen är inte kvar i slutliga koden.

Jag kollar:
- Antalet ratings per rating för att se hur det ser ut mha plot_num_per_rating()  
- Mean rating på de tio i topp filmerna mha plot_bar()
- För att se extremer gör jag en Scatterplot för rating per userID mah scatter_plot(df)
  Den ligger ner för att få bättre översyn, y-axeln är userID och inte intressanta i sig.
  Så det gör inget att det är ett gytter.
- Efter diverse tester med olika sätt att filtrera kollar jag hur totalen påverkats mha plot_rating_before_after()


#### Filtering 
1977 gjordes en av de filmer som fått mest röster, även 1991, 93, 94 som är på top 10 listan ska inte sorteras bort
Testar att kolla vilka top 200 filmer det finns map röster
sum_rating_movieId vid 25 testar jag först och då är vi på halva listan i datan och summorna är väldigt låga redan halvvägs i datamängden.

Alla år i datasetet:

Min: 1874, Max: 2018

['1995' '1994' '1996' '1976' '1992' '1988' '1967' '1993' '1964' '1977'
 '1965' '1982' '1985' '1990' '1991' '1989' '1937' '1940' '1969' '1981'
 '1973' '1970' '1960' '1955' '1959' '1968' '1980' '1975' '1986' '1948'
 '1943' '1950' '1946' '1987' '1997' '1974' '1956' '1958' '1949' '1972'
 '1998' '1933' '1952' '1951' '1957' '1961' '1954' '1934' '1944' '1963'
 '1942' '1941' '1953' '1939' '1947' '1945' '1938' '1935' '1936' '1926'
 '1932' '1979' '1971' '1978' '1966' '1962' '1983' '1984' '1931' '1922'
 '1999' '1927' '1929' '1930' '1928' '1925' '1914' '2000' '1919' '1923'
 '1920' '1918' '1921' '2001' '1924' '2002' '2003' '1915' '2004' '1916'
 '1917' '2005' '2006' '1902' '1903' '2007' '2008' '2009' '1912' '2010'
 '1913' '2011' '1898' '1899' '1894' '2012' '1909' '1910' '1901' '1893'
 '2013' '1896' '2014' '2015' '1895' '1911' '1900' '2016' '2017' '2018'
 '1905' '1904' '1891' '1892' '1908' '1897' '1887' '1888' '1890' '1878'
 '1874' '1907' '1906' '1883']
 
 Jag kollade upp hur mycket som sorteras bort med: 
 - alla filmer som har färre än 48 ratings
 - sen alla filmer med e medelrating under 3
 - sen alla users som röstat på färre än 15 filmer.


Jämförde sedan med att ta bort alla sum_rating under 5000 och jag ser ingen stor skillnad. 
Filtering av 'sum_rating_movieId' i def filter_on_sum_ratings(df) var det som blev den beslutade filteringen.

- Ratingskalan går från 0 till 5 i 0.5 steg
	
- Alla år kunde delta. Men det finns en sortering för EDA scriptet. 
  Gjorde en för att kunna se vad datorn klarade och få en årsfiltering som sedan inte behövdes


#### 1.1 b) Vilka är de 10 filmerna med flest ratings?
Movies with the 10 highest amount of ratings
- Toy Story (1995)
- Braveheart (1995)
- Star Wars: Episode IV - A New Hope (1977)
- Pulp Fiction (1994)
- Shawshank Redemption, The (1994)
- Forrest Gump (1994)
- Jurassic Park (1993)
- Schindler's List (1993)
- Silence of the Lambs, The (1991)
- Matrix, The (1999)
				  
				  
#### 1.1.c  Beräkna den genomsnittliga ratingen för dessa 10 filmerna med flest ratings.
Mean rating on all 10 top movies 4.1

#### 1.1 d - f. svaren finns i scriptet exploratory_data_analysis.py samt bilderna på plottarna finns i powerpointen. 
De flesta plottar måste vara i stor storlek för att kunna bedömmas.


## Recommender in recommender.py

### sklearn coo_matrix
Den är snabb men hittar inte många filmer som pandas Pivot_table gör.

Exempel på vad den inte hittar: 
- toy story - nej, allt tomt index visas och titlar finns för dessa 
- bravehart - ja
- Notting Hill -ja
- Coccer - inga rekommendationer men hittar (Shaolin Soccer (Siu lam juk kau) (2001) Index:  6509)
- Loner - hittar sökt film men tomt i listan utöver index (alltså inga titlar visas men de finns)
- Men in black - hittar sökt film men tomt i listan utöver index (alltså inga titlar visas men de finns)

Exempel på lyckad sökning:

Movie selected:  Matrix, The (1999)

Searching for recommendations.....
Output:
- 'Matrix, The (1999)'
- 'Halloween III: Season of the Witch (1982)'
- 'Double Jeopardy (1999)'
- 'Amityville Horror, The (1979)'
- 'Men in Black (a.k.a. MIB) (1997)'

En notis: 
Slår jag bara Enter vid sökning via user_input så söker de efter Toy story då den har index 0. 
men coo_matrix hittar inte Toy Story (oavsett Enter eller söker på nman toy, toy story, Toy Story).
Mina funktioner check_result() och check_result_index() är ett resultat av falleringen med coo_matrix då jag trodde jag gjort fel i filtereringen.



### Pandas pivot crs_matrix
Den hittar det jag söker tex toy story mfl, inte lika många fel jämfört med coo_matrix. Stabil map sökningsförmåga.
Om rekommendationerna är bra eller inte, vet jag inte då jag inte tittar på film. Visst, vissa kan jag förstå men rekommendationerna verkar så där.


#### Funktion: recommender_dataframe(data, model, n_recommendations)
Så här fungera funktionen jag gjort:
Så länge som True gäller kan sökning på nya filmer fortsätta.
Användare tillfrågas om Titel på film som den vill ha rekommendationer för. 
Input ges till variablen movie_name.
Om längden på inputen är över noll är användaren kvar i loopen.

Med model.fit så beräknas mean och std mha NearestNeighbors().

idx = process.extractOne(movie_name, df_movies['title'])[2]
process.extractOne returnerar en tuple på den högsta string match på inputen den får. 
Den får den titeln på filmen användaren angett, vilken dataframe kolumn den ska titta i, där [2] ger index. 
Retur från extractOne utan [2] är: {tuple3}('Toy Story (1995)', 90, 0) där sista elementet är index

Sedan printas vilken filmtitel som är vald av användaren.
Movie selected:  Toy Story (1995) 
därefter en print som säger att sökning pågår.
Mha NearestNeighbors() så räknas minsta avståndet ut (NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=6))
modellen ges max neighbors  = 6 (den räknar med den sökta så därför +1 från begärda 5 i uppgiften)
och dataframe får filmens index. Retur från modellen är indices, distances.
Indices innehåller filmerna index, distances innehåller avstånden

Därefter printas titlarna för indexarna i indices[0] , 0 för att det är en nestlad lista och jag vill loopa den innersta
När alla printats så meddelar jag hur man avbryter sökningen
Vid fel input så avbryts allt
vid ValueError avbryts allt



#### Funktion: recommender(data, model, n_recommendations)
Så här fungera funktionen jag gjort:

Den fungerar ganska likadant som ovanstående med vissa undantag:

Så länge som True gäller kan sökning fortsätta.
Användare tillfrågas om titel på film som den vill ha rekommendationer för. 
Input ges till variablen movie_name. Om längden på input är över noll är användaren kvar i loopen.

idx = process.extractOne(movie_name, df_movies['title'])[2]
process.extractOne returnerar en tuple på den högsta string match på inputen den får. 
Den får den film användaren angett, vilken dataframe kolumn den ska titta i, där [2] ger index.
Retur från extractOne utan [2]: {tuple3}('Toy Story (1995)', 90, 0) där sista elementet är index

coo_matrix supportar inte indexing så därför måste man spara över till ett annat sparse format via .tocsr och då kan man indexera med idx
data_index = data.tocsr()[idx:]

Med model.fit så beräknas mean och std mha NearestNeighbors().
sen printas vald film 
print av att sökning pågår

sedan mha modellen* så räknas minsta avståndet ut * (NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=6))
Modellen ges max neighbors  = 6 (den räknar med den sökta så därför +1 från begärda 5 i uppgiften)
och dataframe får filmens index, den returnerar indices, distances. Där indices innehåller filmerna index, distances innehåller avstånden.

Därefter printas titlarna för indexarna i indices[0] , 0 för att det är en nestlad lista och jag vill loppa den innersta listan.
Därefter fick hämta values och rensa titlarna då returen från coo_matrix såg annorlunda ut jmft med pd.pivot_table
Print av titlarna som hittats:
När alla printats så meddelar jag hur man avbryter sökningen
Vid fel input så avbryts allt
vid ValueError avbryts allt



## Hur en rekommendations motor fungerar
En rekommendationsmotor använder en kombination av data och maskininlärnings teknologi och är en subklass inom informationsfiltreringssystemen. Den försöker förutse betyget eller preferensen som en användare ska ge produkten. Ju mer data, desto mer effektiv blir den på att lämna relevanta inköpsgenererande förslag.

En rekommendations motor är en typ av datafiltrerare som använder maskininlärningsalgorithmer för att rekommendera den data som är mest relevant för användaren.
Den söker efter mönster i användarens data på dess beteende och denna datan kan hämtas in tydligt(explicit) och otydligt(implicit).
Anledningden till att använda rekommendations motor är att man vill öka försäljning, bibehålla kunder och leverera mer personifierade kundupplevelser.

Det finns tre huvudtyper av rekommendations motor 
- kollaborative filtering
- innehållsbaserad filtering
- hyrid av dessa två


### Kollaborativ filtrering
Kollaborativ filtrering fokuserar på att samla och analysera data från användarbeteende, aktiviteter för att förutspå vad personen gillar, baserat på likheten av vad andra gillar.
För att plotta och beräkna dessa likheter använder kollaborativ filtering en matris formel.
En fördel med kollaborativ motor är att den behöver inte analysera eller förstå innehållet (filmer, böcker, motorsågar(produkter). Den rekommenderar helt enklet på bas av vad de vet om användaren.

Det finns två huvudspår för kollaborativ filtrering.
Modelbaserad: här tränas en modell att lära sig användares representationer från användarinteraktionsmatrisen. 
Tidigare användare ger data(finns i matrisen) och då kollas det om mönstret i insamlade datan stämmer med nuvarande användare.

Minnesbaserad: förlitar sig på likheter från användaren eller produkten. Minnesbaserad i sig delas upp i två spår.
- Hitta användare med liknande intressen (highest similiarity score) och rekommendera produkter som liknande användare röstat högst på.
- Rekommenderar produkter som användaren köpt eller produkter användaren röstat högst på och rekommenderar liknande produkter
  Beräkna similarity score är i verkligheten mer kompliserad än man kan tro. Det beror på en stor andel produkter på varje online platform och datans strukturella   
  skillnader (diversity). Tex så kan det finna ninär data, gilla, ogilla. Men samtidigt kollar man på rating, antal klick, tid som spenderats på sidan etc.
K-Nearest Neighbors är en populär och enkel maskininlärningsalgoritm och är standard lösningen på sådan här problem.
		

### Innehållsbaserad filterering
Innehållsbaserad filterering fungerar på sätt, att om du gillar en viss sak så kommer du också att gilla den andra saken. För att göra rekommendationer, använder algoritmerna en användarprofils preferenser och en beskrivning av produkten (genre, produkttyp, färg, ordlängd), för att komma fram till likheter så används cosine och euklidiska distanser.
Nackdelen med innehållsbaserad filtrering är att den är begränsad till att rekommendera liknande produkter som personer redan köper eller använder. Den kan inte utöka och rekommendera andra typer av produkter eller innehåll. Tex så kan den inte rekommendera kunden andra produkter om kunden bara köpt hushållsmaskiner.

### Hybrid model
En hybrid rekommendationsmotor tittar på både meta (collaborative) data och transaktions data (innehållsbaserade). Pga av detta så överträffas både kollaborative - och innehållsbaserad motor.


En rekommendationsmotor går igenom fyra steg.
#### 1. Datainsamling
Den första och viktigast steget är datainsamling. Det är två huvudtyper av data som ska samlas in.
Implicit data: Information från aktiviteter tex internethistorik, klick, cart events, söklogg, orderhistorik
Explicit data: Information från användaresinmatning, så som recension och betyg, likes och dislikes, produktkommentarer.

Den använder också användareegenskaper såsom åler, kön, och psykologiska intressen och värderingar för att identifiera liknande användare samt funktionsdata tex genre, föremålstyp för att identifiera produktlikhet.

#### 2. Datalagring
När väl datan är inhämtad så  måste den lagras, med tiden kommer mängden data att växa enormt. Det betyder att man måste finnas gott om skalbar lagring. Beroden på vilken typ av data som ska lagras finns olika lagringar tillgängliga.

#### 3. Dataanalys
Det finns flera olika sätt att analysera data. Ett som är klart är att man måste dyka ner djupt och analysera datan
Tex 
Realtidanalys: Datan processas när den skapas
Batchanalys: Datan processas periodvis
Near-real-time: Datan processas per minut iställer för per sekund då den inte behövs omedelbart.

#### 4. Datafiltrering
Sista steget är filterering. Det används olika matriser och matematiska regler vars val beror på om datan som 
filtreras är kollaborativ, innehållsbaserad eller en hybridmix. Slutligen kommer det ut rekommendationer från det här. 

Länkar:

https://www.digitalscaler.eu/blog/digital-data/how-do-recommendation-systems-work/

https://www.appier.com/blog/what-is-a-recommendation-engine-and-how-does-it-work/

https://builtin.com/data-science/recommender-systems
