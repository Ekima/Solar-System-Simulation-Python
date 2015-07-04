# -*- coding: utf-8 -*-

# Ugeopgave for uge 2 af Sarah Vang Nøhr.
# Simulering af solsystemet, planeternes baner.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import sys
from copy import deepcopy

# *** FORMATERING AF NASA JPL EPHEMERIS FORM ***
# --------------------------------------------------------------------
# Kommasepareret ren tekstfil, hvor hvert tidspunkt er
# linjesepareret, indholdende følgende elementer pr.
# tidspunkt, i rækkefølge:
# Datoangivelse i Epoch Julian Date-format
# Koordinattidspunkt (CT) i standard dato-tid format
# Position i X, Y, Z format
# Hastighedsvektor i X, Y, Z format
#
# Eksempel på linje:
# 2456293.500000000, A.D. 2013-Jan-01 00:00:00.0000, -1.284282111761733E-03, -2.455154076901959E-03, -4.207238483437137E-05,  6.145761414237818E-06, -1.318053417580493E-06, -1.306280382523269E-07,


# Variable til brug af udregninger.
# Planeternes masse gange gravitationskonstanten
GMSolen = 2.959122082322128 * 10 ** -4
GMMerkur = 4.912549571831092 * 10 ** -11
GMVenus = 7.243453179939512 * 10 ** -10
GMJorden = 8.887692546888129 * 10 ** -10
GMMars = 9.549531924899252 * 10 ** -11
GMJupiter = 2.824760453365182 * 10 ** -7
GMSaturn = 8.457615171185583 * 10 ** -8
GMUranus = 1.291894922020739 * 10 ** -8
GMNeptun = 1.524040704548216 * 10 ** -8
GMPluto = 1.945211846204988 * 10 ** -12

class solsystem:
    """
    Klasse, der repræsenterer en simulationstilstand af et solsystem.
    """
    planeter = []
    start = 0
    
    def __init__(self, planeter = None):
        """
        Konstruerer et solsystem.
        
        solsystem()
        Opret et standard solsystem.
        (PreConditions)
        Der skal i samme mappe som programmet være følgende
        filer med data i NASA JPL Ephemeris form, som beskrevet ovenfor:
        Sun.txt     Mercury.txt     Venus.txt       Earth.txt       Mars.txt
        Jupiter.txt     Saturn.txt      Uranus.txt      Neptune.txt     Pluto.txt
        (PostConditions)
        Uddata: Et solsystem bestående af 10 planeter, hvoraf den første er
        solen, hvor hver planet indeholder dataen fra hver af de angivne filer.
        
        solsystem(planeter)
        Opret et brugerdefineret solsystem.
        (PreConditions)
        Inddata: planeter - En liste med et eller flere unikke elementer af typen planet.
        Der tages ikke hensyn til, om planeterne har matchende JDer.
        Første planet skal have minimum en JD, og alle planeterne skal som
        minimum have et element i både positioner og hastigheder.
        (PostConditions)
        Uddata: Et solsystem bestående af elementerne fra listen. Det første
        element i listen vil blive brugt som "sol"/dominerende stjerne.
        """
        self.planeter = []
        if planeter is not None:
            self.planeter = planeter
        else:
            self.planeter.append(planet('Sun.txt', GMSolen))
            self.planeter.append(planet('Mercury.txt', GMMerkur))
            self.planeter.append(planet('Venus.txt', GMVenus))
            self.planeter.append(planet('Earth.txt', GMJorden))
            self.planeter.append(planet('Mars.txt', GMMars))
            self.planeter.append(planet('Jupiter.txt', GMJupiter))
            self.planeter.append(planet('Saturn.txt', GMSaturn))
            self.planeter.append(planet('Uranus.txt', GMUranus))
            self.planeter.append(planet('Neptune.txt', GMNeptun))
            self.planeter.append(planet('Pluto.txt', GMPluto))
        self.start = self.planeter[0].JD[0]
        
    def initJD(self):
        """
        Laver nye lister til Epoch Julian Datoer for alle planeterne.
        (PostCondition)
        Alle planeter i solsystemet har opdateret deres liste af JDer, så den
        består af en liste med et element med værdien af solens første JD.
        """
        for planet in range(0, len(self.planeter)):
            self.planeter[planet].JD = [self.start]
    
    def initPos(self):
        """
        Laver nye lister til positionsvektorer for alle planeterne.
        (PostCondition)
        Alle planeter i solsystemet har opdateret deres liste positioner, så
        den består af en liste med et element med værdien af planetens
        første position.
        """
        posarray = []
        for planet in range(0, len(self.planeter)):
            posarray.append([[self.planeter[planet].positioner[0][0], self.planeter[planet].positioner[0][1], self.planeter[planet].positioner[0][2]]])
        return posarray
    
    def initHas(self):
        """
        Laver nye lister til hastighedsvektorer for alle planeterne.
        (PostCondition)
        Alle planeter i solsystemet har opdateret deres liste hastigheder, så
        den består af en liste med et element med værdien af planetens
        første hastighed.
        """
        hasarray = []
        for planet in range(0, len(self.planeter)):
            hasarray.append([[self.planeter[planet].hastigheder[0][0], self.planeter[planet].hastigheder[0][1], self.planeter[planet].hastigheder[0][2]]])
        return hasarray
    
    def tilstand(self, dato):
        """
        Returnerer en liste med alle planeters position for en given dato,
        hvis denne findes for planeten.
        (PostCondition)
        Input: dato - Dato i Epoch Julian Date form.
        (PostCondition)
        Output: En liste med positioner i ndarray-form.
        Hvis datoen ikke findes kastes en fejl.
        """
        i = self.planeter[0].JD.index(dato)
        liste = []
        for planet in range(0, len(self.planeter)):
            liste.append(self.planeter[planet].positioner[i])
        return liste
    
    def toLegeme(self, position):
        """
        Beregner accelerationsvektoren for en position ved brug af 2-legeme modellen,
        ud fra solens seneste positionsvektor.
        (PreCondition)
        Input: position - Positionsvektor i ndarray-form.
        (PostCondition)
        Output: Accelerationsvector i ndarray-form.
        """
        afstand = position - self.planeter[0].positioner[len(self.planeter[0].positioner)-1]
        a = -((self.planeter[0].GM / ((np.sqrt(afstand.dot(afstand))) ** 3)) * afstand)
        return a
        
    def nLegeme(self, positioner):
        """
        Beregner accelartionsvektorer for alle planeter ved brug N-legeme modellen.
        (PreCondition)
        Input: positioner - Liste af n positionsvektorer i ndarray-form. Listen skal være
        ordnet, så positionsvektoren er i samme rækkefølge som den ønskes ift.
        planeterne i solsystemet.
        (PostCondition)
        Output: Returnerer en liste af n længde med accelerationsvektorer i
        ndarray-form i samme ordnede række som ved input.
        """
        alist = []
        for planeten in range(0, len(self.planeter)):
            ai = np.array([0, 0, 0])
            for planet in range(0, len(self.planeter)):
                if planet <> planeten:
                    afstand = np.array([positioner[planeten]]) - np.array([positioner[planet]])
                    a = -((self.planeter[planet].GM / ((np.sqrt(afstand[0].dot(afstand[0]))) ** 3)) * afstand[0])
                    a = np.array(a)
                    ai = np.add(ai, a)
            if ai.all() <> np.array([0, 0, 0]).all():
                alist.append(ai)
        return alist
    
    def euler(self, slut, skridt, detaljeret = False):
        """
        Simulerer solsystemets udvikling ud fra de indtastede data, med
        udgangspunkten i den første dato i solens JDer, ved brug af
        Eulers Integrationsmetode.
        
        euler(slut, skridt)
        Simulationen sker ved hjælp af 2-legeme modellen.
        (PreConditions)
        Alle planeter skal have minimum en gyldig hastigheds- og positionsværdi,
        solen skal desuden have en gyldig JD-værdi.
        Inddata: slut - Numerisk værdi, dato (i Epoch Julian form) hvor
        simulationen skal slutte.
        Inddata: skridt - Positiv numerisk værdi, angiver længden på hvert
        simulationsskridt i tid
        (PostConditions)
        Solsystemets planeter er opdateret med nye værdier af JD, hastigheder
        og positioner, som udregnes ud fra solens første JD, samt første hastigheds-
        og positionsværdier for planeterne.
        
        euler(slut, skridt, detaljeret)
        Simulationen sker ved hjælp af N-legeme modellen, hvis detaljeret
        evaluerer til True, ellers simuleres ved 2-legeme modellen.
        (PreConditions)
        Alle planeter skal have minimum en gyldig hastigheds- og positionsværdi,
        solen skal desuden have en gyldig JD-værdi.
        Inddata: slut - Numerisk værdi, dato (i Epoch Julian form) hvor
        simulationen skal slutte.
        Inddata: skridt - Positiv numerisk værdi, angiver længden på hvert
        simulationsskridt i tid
        Inddata: detaljeret - Sandhedsværdi, skal være True, hvis der ønskes
        at bruges N-legeme modellen, ellers anvendes 2-legeme modellen.
        (PostConditions)
        Solsystemets planeter er opdateret med nye værdier af JD, hastigheder
        og positioner, som udregnes ud fra solens første JD, samt første hastigheds-
        og positionsværdier for planeterne.
        """
        posarray = self.initPos()
        hasarray = self.initHas()
        self.initJD()
        i = 0
        start = self.start
        if detaljeret:
            # Start på N-legeme del
            while start <= slut:
                pliste = []
                for planet in range(0, len(self.planeter)):
                    pliste.append(posarray[planet][i])
                alist = self.nLegeme(pliste)
                for planet in range(0, len(self.planeter)):
                    a = alist[planet]
                    vektorer = eulerPos(np.array([posarray[planet][i]]), np.array([hasarray[planet][i]]), a, skridt)
                    r = vektorer[0]
                    v = vektorer[1]
                    posarray[planet].append([r[0][0], r[0][1], r[0][2]])
                    hasarray[planet].append([v[0][0], v[0][1], v[0][2]])
                    self.planeter[planet].JD.append(start + skridt)
                i += 1
                start += skridt
        else:
            # Start på 2-legeme del
            while start <= slut:
                solpos = np.array([posarray[0][i]]) + np.array([hasarray[0][i]])
                posarray[0].append([solpos[0][0], solpos[0][1], solpos[0][2]])
                hasarray[0].append([hasarray[0][i][0], hasarray[0][i][0], hasarray[0][i][0]])
                self.planeter[0].JD.append(start + skridt)
                for planet in range(1, len(self.planeter)):
                    a = self.toLegeme(posarray[planet][i])
                    vektorer = eulerPos(np.array([posarray[planet][i]]), np.array([hasarray[planet][i]]), a, skridt)
                    r = vektorer[0]
                    v = vektorer[1]
                    posarray[planet].append([r[0][0], r[0][1], r[0][2]])
                    hasarray[planet].append([v[0][0], v[0][1], v[0][2]])
                    self.planeter[planet].JD.append(start + skridt)
                i += 1
                start += skridt
        for planet in range(0, len(self.planeter)):
            self.planeter[planet].positioner = np.array(posarray[planet])
            self.planeter[planet].hastigheder = np.array(hasarray[planet])
            
    def midtpunkt(self, slut, skridt, detaljeret = False):
        """
        Simulerer solsystemets udvikling ud fra de indtastede data, med
        udgangspunkten i den første dato i solens JDer, ved brug af
        Midtpunktsmetoden.
        
        midtpunkt(slut, skridt)
        Simulationen sker ved hjælp af 2-legeme modellen.
        (PreConditions)
        Alle planeter skal have minimum en gyldig hastigheds- og positionsværdi,
        solen skal desuden have en gyldig JD-værdi.
        Inddata: slut - Numerisk værdi, dato (i Epoch Julian form) hvor
        simulationen skal slutte.
        Inddata: skridt - Positiv numerisk værdi, angiver længden på hvert
        simulationsskridt i tid
        (PostConditions)
        Solsystemets planeter er opdateret med nye værdier af JD, hastigheder
        og positioner, som udregnes ud fra solens første JD, samt første hastigheds-
        og positionsværdier for planeterne.
        
        midtpunkt(slut, skridt, detaljeret)
        Simulationen sker ved hjælp af N-legeme modellen, hvis detaljeret
        evaluerer til True, ellers simuleres ved 2-legeme modellen.
        (PreConditions)
        Alle planeter skal have minimum en gyldig hastigheds- og positionsværdi,
        solen skal desuden have en gyldig JD-værdi.
        Inddata: slut - Numerisk værdi, dato (i Epoch Julian form) hvor
        simulationen skal slutte.
        Inddata: skridt - Positiv numerisk værdi, angiver længden på hvert
        simulationsskridt i tid
        Inddata: detaljeret - Sandhedsværdi, skal være True, hvis der ønskes
        at bruges N-legeme modellen, ellers anvendes 2-legeme modellen.
        (PostConditions)
        Solsystemets planeter er opdateret med nye værdier af JD, hastigheder
        og positioner, som udregnes ud fra solens første JD, samt første hastigheds-
        og positionsværdier for planeterne.
        
        """
        posarray = self.initPos()
        hasarray = self.initHas()
        self.initJD()
        i = 0
        start = self.start
        if detaljeret:
            # Start på N-legeme del
            while start <= slut:
                pliste = []
                for planet in range(0, len(self.planeter)):
                    pliste.append(posarray[planet][i])
                alist = self.nLegeme(pliste)
                aestlist = []
                for planet in range(0, len(self.planeter)):
                    a = alist[planet]
                    vektorer = eulerPos(np.array([posarray[planet][i]]), np.array([hasarray[planet][i]]), a, skridt)
                    r = vektorer[0]
                    aestlist.append([r[0][0], r[0][1], r[0][2]])
                aestlist = self.nLegeme(aestlist)
                for planet in range(0, len(self.planeter)):
                    a = aestlist[planet]
                    vektorer = eulerPos(np.array([posarray[planet][i]]), np.array([hasarray[planet][i]]), a, skridt)
                    r = vektorer[0]
                    v = vektorer[1]
                    posarray[planet].append([r[0][0], r[0][1], r[0][2]])
                    hasarray[planet].append([v[0][0], v[0][1], v[0][2]])
                    self.planeter[planet].JD.append(start + skridt)
                i += 1
                start += skridt
        else:
            # Start på 2-legeme del
            while start <= slut:
                solpos = np.array([posarray[0][i]]) + np.array([hasarray[0][i]])
                posarray[0].append([solpos[0][0], solpos[0][1], solpos[0][2]])
                hasarray[0].append([hasarray[0][i][0], hasarray[0][i][0], hasarray[0][i][0]])
                self.planeter[0].JD.append(start + skridt)
                for planet in range(1, len(self.planeter)):
                    a = self.toLegeme(posarray[planet][i])
                    vektorer = eulerPos(np.array([posarray[planet][i]]), np.array([hasarray[planet][i]]), a, skridt)
                    r = vektorer[0]
                    aest = self.toLegeme(r[0])
                    pos = np.array([posarray[planet][i]])
                    has = np.array([hasarray[planet][i]])
                    v = np.add(has, (0.5 * (np.add(a, aest)) * skridt))
                    r = np.add(pos, (0.5 * (np.add(has, v)) * skridt))
                    posarray[planet].append([r[0][0], r[0][1], r[0][2]])
                    hasarray[planet].append([v[0][0], v[0][1], v[0][2]])
                    self.planeter[planet].JD.append(start + skridt)
                i += 1
                start += skridt
        for planet in range(0, len(self.planeter)):
            self.planeter[planet].positioner = np.array(posarray[planet])
            self.planeter[planet].hastigheder = np.array(hasarray[planet])

class planet:
    """
    Klasse, der repræsenterer en simulationstilstand af en planet.
    """
    JD = []
    positioner = []
    hastigheder = []
    GM = 0
    
    def __init__(self, sti = None, GM = 0):
        """
        Konstruerer en planet.
        
        planet()
        Opret en tom planet.
        (PreConditions)
        (PostConditions)
        Uddata: En planet uden angivne JD, positioner, hastigheder eller GM.
        
        planet(sti)
        Opret en planet med data fra en fil uden angivelse af gravitiationskonstant.
        (PreConditions)
        Inddata: sti - String med et stinavn til en fil indeholdende data i
        NASA JPL Ephemeris form, som beskrevet ovenfor.
        (PostConditions)
        Uddata: En planet indholdende data for hvert tidspunkt angivet i filen,
        men uden GM.
        
        planet(sti, GM)
        Opret en planet med data fra en fil og med en gravitationskonstant.
        (PreConditions)
        Inddata: sti - String med et stinavn til en fil indeholdende data i
        NASA JPL Ephemeris form, som beskrevet ovenfor.
        Inddata: GM - Numerisk værdi med planetens ønskede gravitationskonstant.
        (PostConditions)
        Uddata: En planet indholdende data for hvert tidspunkt angivet i filen,
        samt GM.
        """
        self.JD = []
        self.GM = GM
        self.positioner = []
        self.hastigheder = []
        poslist = []
        haslist = []
        if sti is not None:
            try:
                input = np.genfromtxt(sti, delimiter = ",")
                print "Filen " + sti + " er blevet indlæst."
            except:
                print "Filen " + sti + " kan ikke indlæses. Planeten er konstrueret, men indeholder ingen dato-, positions- eller hastighedsdata."
                return
            try:
                for l in range(0, len(input)):
                    self.JD.append(input[l][0])
                    poslist.append([input[l][2], input[l][3], input[l][4]])
                    haslist.append([input[l][5], input[l][6], input[l][7]])
                self.positioner = np.array(poslist)
                self.hastigheder = np.array(haslist)
            except:
                print "Filen " + sti + " er ikke af gyldig format. Planeten er konstrueret, men indeholder ingen dato-, positions- eller hastighedsdata."
                return
    
    def printInfo(self):
        """
        Printer den den nuværende simulationstilstand.
        (PreConditions)
        Der skal være samme mængde JD som positioner.
        (PostConditions)
        Print af linjer med Epoch Julian Date og position
        for alle instanser af planeten gemt på nuværende tidspunkt.
        """
        for l in range(0, len(self.JD)):
            print "Epoch Julian Date: " + str(self.JD[l])
            print "Position: " + str(self.positioner[l])

def lavPlot(titel, solsystem, farver, navne):
    """
    Laver et 3-dimensionelt og et 2-dimensionelt plot af et solsystems simulation.
    (PreCondition)
    Inddata: titel - String indeholdende information om simulationen.
    Inddata: solsystem - solsystem Den solsystemssimulation, der ønskes at plottes data fra
    Inddata: farver - Liste af farver, der overholder konventionerne i PyPlot
    Inddata: navne - Liste af strings med navne til graferne (Planetnavne)
    (PostCondition)
    Der er dannet to plots som subplots med den indtastede data i hhv. 3D og 2D,
    som kan vises ved hjælp af plt.show().
    """
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    for planet in range(0, len(solsystem.planeter)):
        r = solsystem.planeter[planet].positioner
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=farver[planet], label=navne[planet])
    plt.xlabel(u"x (AU)")
    plt.ylabel(u"y (AU)")
    plt.ylabel(u"z (AU)")
    plt.title(u"3D-simulering af solsystemet: " + titel)
    plt.grid()
    plt.legend(loc=1, prop={'size':7})
    ax2 = fig.add_subplot(212)
    for planet in range(0, len(solsystem.planeter)):
        r = solsystem.planeter[planet].positioner
        plt.plot(r[:, 0], r[:, 1], color=farver[planet], label=navne[planet])
    plt.xlabel(u"x (AU)")
    plt.ylabel(u"y (AU)")
    plt.ylabel(u"z (AU)")
    plt.title(u"2D-simulering af solsystemet: " + titel)
    plt.grid()
    plt.legend(loc=1, prop={'size':7})
    
def eulerPos(pos, has, a, skridt):
    """
    Udregner næste positionsvektor og hastighedsvektor for et punkt vha. Eulers metode.
    (PreCondition)
    Inddata: pos - Tredimensionelt ndarray indeholdende en positionsvektor
    Inddata: has - Tredimensionelt ndarray indeholdende en hastighedsvektor
    Inddata: a - Tredimensionelt ndarray indeholdende en accelerationsvektor
    Inddata: skridt - Numerisk værdi indeholdende det tidsskridt, der ønskes at udregnes for
    (PostCondition)
    Uddata: Liste med 2 tredimensionelle ndarrays, hvor r angiver den nye positionsvektor
    og v angiver den nye hastighedsvektor.
    """
    r = np.add(pos, has * skridt)
    v = np.add(has, a * skridt)
    return [r, v]

def afvigelse(solsystem0, solsystem1, skridt, titel, farver, navne):
    """
    Returnerer en graf med data omkring, hvor lang afstand, der er mellem hver planet på
    samme plads på samme tidspunkt (JD) i to solsystemssimulationer med samme (antal) planeter.
    (PreCondition)
    Inddata: solsystem0 - solsystem med n antal planeter
    Inddata: solsystem1 - solsystem med n antal planeter
    Inddata: skridt - Numerisk værdi, mindste skridtværdi brugt (i begge solsystemer)
    Inddata: titel - String indeholdende information om simulationsafvigelsen
    Inddata: farver - Liste af farver, der overholder konventionerne i PyPlot
    Inddata: navne - Liste af strings med navne til graferne (Planetnavne)
    (PostCondition)
    Der er dannet et 2D-plot med afvigelsesinformation, som kan vises ved hjælp af plt.show().
    """
    afvliste = []
    a = 0
    b = 0
    while a < len(solsystem0.planeter[0].JD) and b < len(solsystem1.planeter[0].JD):
        s = solsystem0.planeter[0].JD[a]
        t = solsystem1.planeter[0].JD[b]
        if s == t:
            p = solsystem0.tilstand(s)
            r = solsystem1.tilstand(t)
            afv = []
            for i in range(0, len(p)):
                afstand = np.subtract(np.array(p[i]), np.array(r[i]))
                dist = np.sqrt(afstand.dot(afstand))
                afv.append(dist)
            afvliste.append([s, afv])
            a += skridt
            b += skridt
        elif s < t:
            a += skridt
        else:
            b += skridt
    liste = []
    for i in range(0, len(afvliste[0][1])):
        planetdato = []
        planetpos = []
        for n in range(0, len(afvliste)):
            planetdato.append(afvliste[n][0])
            planetpos.append(afvliste[n][1][i])
        liste.append([planetdato, planetpos])
    fig = plt.figure()
    for planet in range(0, len(liste)):
        r = liste[planet]
        r1 = np.array(r[0])
        r2 = np.array(r[1])
        plt.plot(r1, r2, color=farver[planet], label=navne[planet])
    plt.xlabel(u"Epoch Julian Date")
    plt.ylabel(u"Længde i AU (Astronomiske enheder)")
    plt.title(u"Afvigelse: " + titel)
    plt.grid()
    plt.legend(loc=1, prop={'size':7})

def main(args):
    """
    Main funktion med simuleringer og test af funktionerne.
    """
    farver = ["GoldenRod", "GreenYellow", "DeepPink", "ForestGreen", "FireBrick", "DodgerBlue", "DimGray", "MediumSeaGreen", "MediumOrchid", "MediumAquaMarine"]
    navne = [u"Solen", u"Merkur", u"Venus", u"Jorden", u"Mars", u"Jupiter", u"Saturn", u"Uranus", u"Neptun", u"Pluto"]
    solsystemet = solsystem()
    test = solsystem()
    
    # Oprettelse af simulationsplots
    # Simulationer fra .txt filerne
    lavPlot(u".txt data", solsystemet, farver, navne)
    # Simulation med Eulers integrationsmetode, 2-legeme
    solsystemet.euler(2456658.500000000, 1)
    lavPlot(u"Euler, 2-legeme", solsystemet, farver, navne)
    afvigelse(solsystemet, test, 1, u'Euler, 2-legeme', farver, navne)
    # Simulation med Eulers integrationsmetode, n-legeme
    solsystemet.euler(2456658.500000000, 1, True)
    lavPlot(u'Euler, N-legeme', solsystemet, farver, navne)
    afvigelse(solsystemet, test, 1, u'Euler, N-legeme', farver, navne)
    # Simulation med midtpunktsmetoden, 2-legeme
    solsystemet.midtpunkt(2456658.500000000, 1)
    lavPlot(u'Midtpunkt, 2-legeme', solsystemet, farver, navne)
    afvigelse(solsystemet, test, 1, u'Midtpunkt, 2-legeme', farver, navne)
    # Simulation med midtpunktsmetoden, N-legeme
    solsystemet.midtpunkt(2456658.500000000, 1, True)
    lavPlot(u'Midtpunkt, N-legeme', solsystemet, farver, navne)
    afvigelse(solsystemet, test, 1, u'Midtpunkt, N-legeme', farver, navne)
    plt.show()
 
if __name__ == '__main__':
    main(sys.argv)