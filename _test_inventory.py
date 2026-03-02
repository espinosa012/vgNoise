import sys
sys.path.insert(0, 'src')
from core.item import BaseItem, ItemType, Inventory

class Sword(BaseItem):
    ITEM_TYPE = [ItemType.WEAPON]
    def __init__(self): super().__init__(name='Espada', weight=3.0)

class Potion(BaseItem):
    ITEM_TYPE = [ItemType.CONSUMABLE]
    def __init__(self): super().__init__(name='Pocion', weight=0.5)

class Key(BaseItem):
    ITEM_TYPE = [ItemType.TOOL]
    def __init__(self): super().__init__(name='Llave', weight=0.1)

bag = Inventory(capacity=5, max_weight=10.0, allowed_types=[ItemType.WEAPON, ItemType.CONSUMABLE])
sword, potion, key = Sword(), Potion(), Key()

print(bag)
print('espada:', bag.add(sword))
print('pocion:', bag.add(potion))
print('llave (bloqueada):', bag.add(key))
print(bag)
print('peso:', bag.current_weight, '| restante:', bag.remaining_weight)
print('slots restantes:', bag.remaining_slots)
bag.drop(sword, 10.0, 20.0)
print('tras drop:', bag)
print('espada state:', sword.state.name)

