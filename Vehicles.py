class Car:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.collisionbox = [[], []]
    def __repr__(self):
        return "car position,velocity and collision limits : ('{}', '{}', {})".format(self.position, self.velocity, self.collisionbox)

    def GetVel(self):
        return self.velocity

    def GetPos(self):
        return self.position

    def updatePos(self):
        ## collisionbox [0]  is before updating position and 1 after updating position
        self.collisionbox[0] = [self.position-2, self.position +2]
        ## position increases by velocity
        self.position += self.velocity
        if self.position >=500:
            self.collisionbox[0] = [0,0]
            self.collisionbox[1] = [0, 0]
        else:
            self.collisionbox[1] = [self.position-2, self.position+2]
    def CollisionBox(self):
        return self.collisionbox
    def updateVeloc(self, inputvel):

        self.velocity = inputvel


class PlayerCar(Car):
    def __init__(self, position, velocity):
        ## getting all position and vel filled as above
        super().__init__(position, velocity)
        self.velchange = 0
    def __repr__(self):
        return "PLAYERCAR position,velocity and collision limits : ('{}', '{}', {})".format(self.position, self.velocity, self.collisionbox)
    def updatePos(self):
        self.collisionbox[0] = [self.position - 2, self.position + 2]## before updating 
        self.position += self.velocity
        self.collisionbox[1] = [self.position - 2, self.position + 2] ## after updating

    def updateVeloc(self, inputvel):
        temp = self.velocity

        self.velocity = inputvel
        self.velchange = self.velocity - temp


class Truck(Car):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.velchange = 0
        self.collisionbox = [[], []]
    def __repr__(self):
        return "TRUCK position,velocity and collision limits : ('{}', '{}', {})".format(self.position, self.velocity, self.collisionbox)
    
    def updatePos(self):
        self.collisionbox[0] = [self.position - 3, self.position + 3]## before updating 
        self.position += self.velocity
        self.collisionbox[1] = [self.position - 3, self.position + 3] ## after updating

    def updateVeloc(self, inputvel):
        temp = self.velocity

        self.velocity = inputvel
        self.velchange = self.velocity - temp

# def main():
#     p = PlayerCar(0,5)
#     t = Truck(0,3)

#     print(isinstance(p,PlayerCar))
#     print(isinstance(p,PlayerCar))
#     print(isinstance(p,Truck))



# if __name__ == "__main__":
#     main()
