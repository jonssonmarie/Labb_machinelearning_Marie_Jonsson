#Labb Maskininlärning

Syftet med den här labben är att använda verktygen ni lärt er i maskininlärningen till att få kännedom och
tillämpa på olika slags problem som ni kan stöta på i näringslivet.
Algoritmerna och modellerna vi bygger upp här kommer vara enkla och i näringslivet är det
inte ovanligt att man kombinerar flertals modeller i sina lösningar. Poängen med den här labben är att få en
förståelse av hur man kan angripa olika slags problem mha maskininlärning.

Delmoment
Den här labben är uppdelad i två delmoment:
- Lab1 :  Recommender system
- Lab 2:  Disease prediction

### Lab 1 Recommender system
När du tittar på Youtube, beställer mat online, köper böcker online, lyssnar på Spotify, använder LinkedIn så
får du ständigt rekommendationer för nya videoklipp, maträtter mm.  
Det som ligger bakom dessa är en typ av rekommenderarsystem.

#### EDA
Ladda ned datasetet besvara på följande frågor:  
a) Gör en EDA för att förstå datasetet.   
b) Vilka är de 10 filmerna med flest ratings?  
c) Beräkna den genomsnittliga ratingen för dessa 10 filmerna med flest ratings.  
d) Gör en plot över årtal och antalet filmer representerade i datasettet.  
e) Gör en plot över antalet ratings mot movieId.  
f) Beräkna genomsnittliga ratings för de top 10 filmerna med flest ratings.   
Gör ett stapeldiagram över dessa.

#### Skapa gles matris
Pivottabell är dock "dyrt" att skapa och förmodligen kommer inte din dator att klara av skapa den om du inte filtrerar bort viss data.  
Fundera ut ett lämpligt sätt att filtrera ditt dataset, pröva dig fram och motivera.  
Skapa en gles (sparse) matris med hjälp av denna pivottabell.

#### Rekommenderarsystemet
Skapa rekommenderarsystemet med KNN och låt systemet ta input från användaren och skriva ut top 5  
rekommenderade filmerna, baserat på användarens sökquery. Notera att det inte gör något för den här  
labben om du tycker rekommendationerna är helt felaktiga, det här systemet vi bygger är alldeles för enkelt.  
a) Beskriv med ord hur ditt system fungerar.  
b) Leta online och läs vidare om rekommenderarsystem och beskriv kort hur dem fungerar. Glöm inte källhänvisa.

### Lab 2. Disease prediction
I det här momentet kommer vi jobba med ett dataset med data för hjärt-kärlsjukdom.  
Börja med att ladda ned datasetet från Kaggle och läs på vad de olika features betyder.   
Notera att detta dataset innehåller många felaktigheter, exempelvis finns negativa blodtryck och blodtryck som är omöjligt höga.

#### EDA uppvärmning
Använd pandas, matplotlib och seaborn för att besvara på följande frågor för datasetet:  
a) Hur många är positiva för hjärt-kärlsjukdom och hur många är negativa?  
b) Hur stor andel har normala, över normala och långt över normala kolesterolvärden? Rita ett tårtdiagram.  
c) Hur ser åldersfördelningen ut? Rita ett histogram.  
d) Hur stor andel röker?  
e) Hur ser viktfördelningen ut? Rita lämpligt diagram.  
f) Hur ser längdfördelningen ut? Rita lämpligt diagram.  
g) Hur stor andel av kvinnor respektive män har hjärt-kärlsjukdom? Rita lämpligt diagram

#### Feature engineering BMI
Skapa en feature för BMI (Body Mass Index), läs på om formeln på wikipedia.  
a) Släng de samples med orimliga BMIer och outliers. Notera att detta kan vara svårt att avgöra i vilket
range av BMIer som vi ska spara. Beskriv hur du gör avvägningen.  
b) Skapa en kategorisk BMI-feature med kategorierna: normal range, overweight, obese (class I), obese
(class II), obese (class III).

#### Feature engineering blodtryck
Släng bort samples med orimliga blodtryck och outliers. Likt uppgift 2.1.0 är det inte trivialt att sätta  
gränserna. Skapa en feature för blodtryckskategorier enligt tabellen i denna artikel.

#### Visualiseringar andel sjukdomar
Skapa barplots med en feature mot andelen positiva för hjärt-kärl sjukdom.  
Exempelvis blodtryckskategorier mot andel positiva, BMI kategori mot andel positiva mm.  
Gör dessa plots i en figur med flera subplots.

#### Visualiseringar korrelation
Skapa en heatmap av korrelationer och se om du hittar features som är starkt korrelerade, dvs nära 1 eller
features som är starkt negativt korrelerade, dvs nära -1.  
Kan du förklara varför de kan vara korrelerade?

#### Skapa två dataset
Skapa en kopia av ditt dataframe. 
- På ena dataframet: ta bort följande features:   
ap_hi, ap_lo, height, weight, BMI. 
gör one-hot encoding på BMI-kategori, blodtryckskategori, kön 
- På andra dataframet: ta bort följande features:  
BMI-kategori, blodtryckskategori, height, weight 
gör one-hot encoding på kön

#### Välja modell
Välj 3-5 maskininlärningsmodeller, gärna så olika som möjligt.  
För varje dataset som vi skapade i uppgift 2.3 gör följande:  
train|validation|test split 
skala datasetet med feature standardization och normalization 
definiera hyperparametrar (param_grids) att testa för varje modell  
använda GridSearchCV() och välja lämplig evalueringsmetric  
gör prediction på valideringsdata 
beräkna och spara evaluation score för ditt valda metric 
checka bästa parametrarna för respektive modell. 
Vilket dataset väljer du och vilken modell väljer du?  
Använd den modellen du valt och träna på hela träningsdatan.

#### Ensemble
Använd VotingClassifier() på datasetet som du valt och lägg in de bästa parametrarna för respektive modell.

#### Evalueringar
Gör confusion matrices och classification reports för 2.4 och 2.5.

#### "Deploy" - spara modell
Börja med att plocka ut 100 slumpmässigt valda rader från ditt dataset.  
Exportera dessa 100 samples i test_samples.csv.  
Därefter tar du den bästa modellen och träna på all data vi har förutom de 100 datapunkterna du plockade ut.  
Spara därefter modellen i en .pkl-fil med hjälp av joblib.dump().  
För modellen kan du behöva använda argumentet compress för att komprimera om filstorleken för stor.

#### Ladda modellen
Skapa ett nytt skript: "production_model.py", ladda in test_samples.csv och din modell.  
Använd joblib.load() för att ladda in en .pkl-fil.  
Gör prediction på de 100 datapunkterna och exportera en fil "prediction.csv" som ska innehålla kolumnerna med ifyllda värden:
- probability class 0
- probability class 1
- prediction
