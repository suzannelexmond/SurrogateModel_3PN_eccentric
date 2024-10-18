import random


bingolijst1 = ['ik hou van techno', 'autist', 'woont nog bij zn ouders', 'meer dan een uur reizen', 'beginnen over het weer', 'psychologiestudent', 'HKU', 'vragen over de bouw', 'opmerking over de vissen', 'elk weekend bij relatie', 'iemand die niet in Utrecht studeert', 'queer', 'ik hou van creatieve dingen', 'oh een wasbak, fijn!', 'minimaal 2 gecancelled']
bingolijst2 = ['diergeneeskundestudent', 'ik ben wel netjes, maar niet heel netjes', 'koorbal/studentenverenigingchapie', 'eerder gehospiteerd op cambridgelaan', 'beginnen over huisroddels', 'international', 'iemand die BNB vol liefde/first dates kijkt', 'neuspiercing', 'iemand die super zenuwachtig is', 'iemand die muziekinstrumenten speelt']
bingolijst3 = ['ik hou van schoonmaken', 'rare verzameling', 'vlees is moord sticker', 'praten over politiek', 'nog een hospi', 'iemand die we al kennen']

def genereer_bingolijst():
    bingolijst = []
    bingolijst.append(random.sample(bingolijst1, 9))
    bingolijst.append(random.sample(bingolijst2, 5))
    bingolijst.append(random.sample(bingolijst3, 2))
    random.shuffle(bingolijst)
    return bingolijst

print(genereer_bingolijst())