import Vehicles
import numpy as np
import math
import cv2
import random
import time
import matplotlib.pyplot as plt 
from PIL import Image
import scipy.misc
from functools import wraps
import copy
import os


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



class GameV1:
    ## 5000 is total length of highway
    ## class attributes
    front  = 20
    back = 20

    # front = 30
    # back = 20
    def __init__(self): 
        #eachunit is 5.2 feet
        #car will be 4 units long
        #


        self.lanes = 5
        self.Game = []
        self.maxUnits = 500
        self.playercarposition = 0
        self.playerlanes = self.lanes-1
        # self.imagearray = np.zeros(shape=(500,600))
        # self.imagearray = np.zeros(shape = (200,300))
        self.imagearray = np.zeros(shape = (200,400))
        self.gameplayerindexvert =0
        self.gameplayerindexhorz =0
        self.give_img = False
        self.temprestore = []
        self.timer = 0
        self.colorImage = None
        self.our_car = []
    
    @classmethod
    def change_climate(cls, climate = "sunny"):
        if climate == "sunny":
            cls.front = 30
            cls.back =  20
        else:
            cls.front = 20
            cls.back =  10
    
    def draw_image(self):
        if (self.give_img == False):
            return
        flag = False
        plt.ion()
        if (not flag):
## checkout some exiciting colors https://matplotlib.org/tutorials/colors/colormaps.html
            ## Paired, Blues, Set1
            figure = plt.imshow(self.imagearray, cmap="Accent")
            flag=True

        else:
            figure.set_data(self.imagearray)
        plt.draw()
        # time.sleep(12)
        plt.waitforbuttonpress(0.01)


    def print_array(self, arr):
        for row in arr:
            for elem in row:
                print(elem, end=' ')
            print()

## prints the positions in the playercarlane


    def visualize_positions(original_func):

        def wrapper(self,*args,**kwargs):
            arr = original_func(self)
            for j in range(len(arr[self.playerlanes])):
                print(arr[self.playerlanes][j], end=" ") 
            return arr
        return wrapper


    # def print_positions_array(self):
    #     all_posn =self.give_me_all_positions()
    #     for j in range(len(all_posn[self.playerlanes])):
    #         print(all_posn[self.playerlanes][j], end=" ")

    # @visualize_positions
    def give_me_all_positions(self):
        all_positions_array = []
        ## for each row in the game array
        for i in range(len(self.Game)):
            all_positions_array.append([])
        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                all_positions_array[i].append(self.Game[i][j].GetPos())
            #     print(self.Game[i][j].GetPos(), end = " ")
            # print()
        return all_positions_array

## collision detection based on positions
    def collision_detection_player(self):
        all_positions_array = self.give_me_all_positions()
        self.searchforPlayerCar()
        player_position  = self.Game[self.playerlanes][self.playercarposition].GetPos()

        length = len(all_positions_array[self.playerlanes])
        for j  in range(0,length):
            if  (j != self.playercarposition):
                pos = all_positions_array[self.playerlanes][j] 
                if abs(int(pos-player_position))<4:
                    # print("pos_that_creates_a_crash_followed_by_player_position",pos, player_position)
                    # print("disable for now")
                    return True
        return False






    def populateGameArray(self):
        self.Game=[]
        self.timer = 0
        # import time
        random.seed(int(time.time()))
        for i in range(self.lanes):
            self.Game.append([])
        for i in range(self.lanes):
            currentposition = 0
            while(currentposition<self.maxUnits):
                ### velocity and position randomized
                if (i==3):
                    currentposition += 50 ## change the freq from 30 to 40                   
                    # self.Game[i].append(Vehicles.Truck(currentposition, np.random.normal(1+(0.3*(self.lanes-i-1)), 0.3)))
                    self.Game[i].append(Vehicles.Truck(currentposition, 1))

                    continue
                    # continue
                
                self.Game[i].append(Vehicles.Car(currentposition, 2 * (self.lanes-i)))
                
                
                
                # self.Game[i].append(Vehicles.Car(currentposition, np.random.normal(2+(1+0.3*(self.lanes-i-1)), 0.3)))

                # currentposition += random.randrange(8,20)## any number between 8 and 20
                # self.Game[i].append(Vehicles.Car(currentposition,2))
                currentposition += 50 ##  change freq from 20 to 30

        self.Game[self.lanes-1][0] = Vehicles.PlayerCar(0, self.Game[self.lanes-1][1].velocity)











    def updateGameArray(self, action):
        self.timer +=1
        self.tempstore = self.Game

        self.searchforPlayerCar()## will set the self.playerlanes and self.carposition
        # print("before",self.playerlanes,self.playercarposition)

        reward = self.updatePlayerCar(action)
        # print("reward",reward)
        self.searchforPlayerCar()## will set the self.playerlanes and self.carposition        
        # print("after",self.playerlanes,self.playercarposition)



        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                updatingcar = self.Game[i][j] ## every car object

                if (isinstance(updatingcar, Vehicles.PlayerCar) == False):
                #     ## if its other car
                #     # if the next car to it is some car probably also the last car in that lane
                #     ## which means updating car should not be the last car
                #     if (j+1 <= len(self.Game[i]) -1):
                #         # if the difference between position of this car and next car is less than 0.3
                #         # if (updatingcar.GetVel() -self.Game[i][j+1].GetVel() <= 4): ## change it to 4 from 0.3
                #             ## give the same velocity as the next adjacent car
                #             updatingcar.updateVeloc(self.Game[i][j+1].GetVel())
                #     ## if the previous car to updating car is the not the first car
                #     if (j-1 != 0):
                #         ## chance of collision
                #         if updatingcar.GetVel() < self.Game[i][j-1].GetVel():
                #             # if self.timer%10 == 0:
                #                 ##
                #                 # updatingcar.updateVeloc(np.random.normal(2+(1+0.3*(self.lanes-i-1)), 0.15))
                #                 updatingcar.updateVeloc(self.Game[i][j-1].GetVel())
                    ## we have to update the position in every case but velocities only for middle cars
                    ## below  is valid for both start and end car updatepos()

                    ## prev car exist
                    # if (j-1>=0):
                    #     if (updatingcar.GetVel()>self.Game[i][j-1].GetVel()):
                    #         updatingcar.updateVeloc(self.Game[i][j-1].GetVel())
                    # updatingcar.updatePos()
                    updatingcar.updatePos()

                    ## if the position is greater than 500
                    if updatingcar.GetPos() >= self.maxUnits:
                        self.Game[i].pop(j) ## remove that jth car object
                        self.Game[i].insert(0, updatingcar)## insert in position 0
                        updatingcar.position = 0
        self.searchforPlayerCar() ## this will fill playerlanes and playercarposition
        playercar = self.Game[self.playerlanes][self.playercarposition]
        playercar.updatePos()## update the player car this will update the game array position and above
        

        #if action ==2 or action ==3:
        #    if self.playercarposition != len(self.Game[self.playerlanes]) -1:
        #        updatingcar.velocity = self.Game[self.playerlanes][self.playercarposition+1].GetVel()
        # print('lanes: {}, action: {}, velocity: {}'.format(self.playerlanes, action, playercar.GetVel()))

        # self.createImage(self.createImageList(), self.gameplayerindexvert, self.gameplayerindexhorz)
        self.image_from_visible_objects()

        # self.render_image()
        # plt.plot(self.imagearray)
        # plt.show()

        # img = Image.fromarray(self.imagearray, 'L')
        # img.show()


        

        # for i in range(200):
        #     for j in range(300):
        #         print(self.imagearray[i][j], end =" ")
        #     print()





        self.searchforPlayerCar()
        if self.checkColission():
        # if self.collision_detection_player():
            # print("crash")
            # plt.imshow(self.imagearray, cmap="Blues")
            # plt.show()
            
        #    commented for now # 
            self.draw_image()
            
            
            return -1, 0
        elif playercar.GetPos() >=self.maxUnits:
            # print("win")
            # plt.imshow(self.imagearray, cmap="Blues")
            # plt.show()
            
        #    commented for now #            
            self.draw_image()
            return 1, reward
        else:
            # print("no_crash")
            # plt.imshow(self.imagearray, cmap="Blues")
            # plt.show()
        #    commented for now #
            # print("something")            
            self.draw_image()
            

            return 0, reward






    def searchforPlayerCar(self):
        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                if isinstance(self.Game[i][j], Vehicles.PlayerCar):
                    self.playerlanes = i
                    self.playercarposition =j
                    break



















    ## return reward function
    def updatePlayerCar(self, action):
        #action 0 = ACC : Adaptive cruise control follow front car
        #action 1 = velocity +=2 accelerate in same lane
        #action 2 = left lane change
        #action 3 = right lane change

        playercar = self.Game[self.playerlanes][self.playercarposition]

        if (action == 0):

            if self.playercarposition != len(self.Game[self.playerlanes]) -1:
                if (self.Game[self.playerlanes][self.playercarposition+1].GetPos()> playercar.GetPos()):
                    playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition+1].GetVel())
                #if playercar.velchange > 0:
                #    return playercar.velchange
                #if playercar.velchange < 0:
                #    return playercar.velchange

                return 0
            else:

                return 0
        elif (action == 1):
            #self.searchforPlayerCar()
            playercar.updateVeloc(playercar.GetVel() + 0.5)
            return 0

        elif (action == 2):

            if self.playerlanes != 0:

                for i in range(len(self.Game[self.playerlanes -1])):

                    if self.Game[self.playerlanes-1][i].GetPos() >= playercar.GetPos():

                        self.Game[self.playerlanes].pop(self.playercarposition)
                        self.Game[self.playerlanes-1].insert(i, playercar)
                        return self.Game[self.playerlanes-1][i+1].GetVel() - playercar.GetVel()
            else:


                if self.playercarposition != len(self.Game[self.playerlanes]) - 1:
                    playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition + 1].GetVel())
                    return 0
        elif (action == 3):

            if self.playerlanes != self.lanes-1:

                for j in range(len(self.Game[self.playerlanes+1])):
                    if self.Game[self.playerlanes+1][j].GetPos() >= playercar.GetPos():
                        self.Game[self.playerlanes].pop(self.playercarposition)
                        self.Game[self.playerlanes+1].insert(j, playercar)
                        return self.Game[self.playerlanes+1][j+1].GetVel() - playercar.GetVel()
            else:
                if self.playercarposition != len(self.Game[self.playerlanes]) -1:
                    playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition+1].GetVel())
                    return 0
        # else:
        #     return 0

    ## if this returns true meaning it has collided
    def checkColission(self):
        playercar = self.Game[self.playerlanes][self.playercarposition]
        if self.checkBack(playercar):
            # print("1")
            return True
        elif self.checkFront(playercar):
            # print("2")
            return True
        elif self.checkFast(playercar):
            # print("3")
            return True
        # elif self.checkCurrent(playercar):
        #     # print("4")
        #     return True
        # elif self.checkTotal(playercar):
            # print("5")
            # return True
        # if self.collision_detection_player():
        #     print("6")
        #     return True
        else:
            return False


    # def checkTotal(self,playercar):
    #     self.searchforPlayerCar()
    #     player_position = playercar.GetPos()
    #     for obj in self.Game[self.playerlanes]:
    #         print(obj.GetPos(),end = " ")
    #         if abs(obj.GetPos() - player_position<=4):
                
    #             return True

    #     return False



    # def checkCurrent(self,playercar):
    #     player_position = playercar.GetPos()
    #     posn_b_player =self.Game[self.playerlanes][self.playercarposition-1].GetPos()
    #     posn_a_player = self.Game[self.playerlanes][self.playercarposition+1].GetPos()

    #     if abs(player_position - posn_b_player)<4:
    #         return True
    #     if abs(player_position- posn_a_player)<4:
    #         return True
    #     return False






    def checkBack(self, playercar):
        if self.playercarposition !=0:
            if playercar.CollisionBox()[1][0] <= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1]:
                # print(playercar.CollisionBox()[1][0])
                #print("playercar")
                #print(self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1])
                #print("car behind")
                return True
        return False



  


    def checkFront(self, playercar):
        self.searchforPlayerCar()
        if self.playercarposition < (len(self.Game[self.playerlanes])-1):

            if playercar.CollisionBox()[1][1] >= self.Game[self.playerlanes][self.playercarposition+1].CollisionBox()[1][0]:
                # print(playercar.CollisionBox()[1][1])
                # print(self.Game[self.playerlanes][self.playercarposition+1].CollisionBox()[1][0])
                return True

        return False



    ## go thorugh in two time stamps
    def checkFast(self,playercar):
        if (self.playercarposition!=0):

            if playercar.CollisionBox()[0][1] <= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[0][0]:

                if playercar.CollisionBox()[1][0] >= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1]:
                    #print("2 fast")
                    return True

        return False

    # def checkLeft(self,playercar):
    #     if (self.playercarposition!=0):

    ## decorator
    # def print_any_array(org_func):
    #     def print_wrapper(*args,**kwargs):
    #         arr = org_func(*args,**kwargs)
    #         for row in arr:
    #             for elem in row:
    #                 print(elem, end=' ')
    #             print()
    #         return org_func()

    #     return print_wrapper

    ##visibility array
    ## self.createImagelist() will fill visibility_array positions
    # @print_any_array
    # def createImageList(self):
    #     playercar = self.Game[self.playerlanes][self.playercarposition]

    #     visibility_array = []

    #     # front = 20
    #     # back=20
    #     ## for each row in the game array
    #     for i in range(len(self.Game)):
    #         visibility_array.append([])
    #     for i in range(len(self.Game)):

    #         for j in range(len(self.Game[i])):

    #             ## behind car visibility by 10
    #             if self.Game[i][j].GetPos() >= playercar.GetPos() - self.back:
    #                 ## if car at i,j is in range p-10 to p
    #                 ## possible values are {0,1,2,3,4,5,6,7,8,9}
    #                 if self.Game[i][j].GetPos() <= playercar.GetPos():
    #                     ## visibility array will have {100,110,120,....190,200}
    #                     visibility_array[i].append((self.Game[i][j].GetPos() - playercar.GetPos()) * 10 + 10*self.back ))
    #                 if self.Game[i][j].GetPos() == playercar.GetPos():

    #                     self.gameplayerindexvert = i
    #                     ## our car will be put in the visibility_array recently
    #                     ## therefore last element of subarray
    #                     self.gameplayerindexhorz = len(visibility_array[i]) - 1 

    #             if self.Game[i][j].GetPos() <= playercar.GetPos() + self.front:
    #                 if self.Game[i][j].GetPos() > playercar.GetPos():
    #                     visibility_array[i].append((self.Game[i][j].GetPos() - playercar.GetPos()) * 10 + 10*self.back)
    #     return visibility_array

## create Imagelist outputs visibility array on vehicle  objects that are visible
## its only array of position and does not have vehicle objects in it

##  create Image
## shade where is final painting 

 

## will fill imagearray
    # def createImage(self, visibility_array, playervert, playerhorz):
    #     self.imagearray = np.zeros(shape=(np.shape(self.imagearray)[0],np.shape(self.imagearray)[1])) ## display image array
    #     self.imagearray.fill(0.1)
    #     for i in range(len(visibility_array)):
    #         for j in range(len(visibility_array[i])):
    #             if i == playervert:
    #                 if j == playerhorz:
    #                     ## for our car spotted
    #                     self.shadewhere(visibility_array[i][j], i, 0.5)
    #                 else:
    #                      ## car in same row but not our car 
    #                     self.shadewhere(visibility_array[i][j], i, 1)
    #             else:
    #                 ## car in different row
    #                 self.shadewhere(visibility_array[i][j], i, 1)



    ## fill the imagearray
    ## the position is for display only not the real position as 
    def shadewhere(self, position, lanenumber, value):
        ## 4 units for the car
        leftmax = int(math.floor(position - 20))
        rightmax = int(math.floor(position + 20))
        if leftmax < 0:
            leftmax = 0
        ## cannot exceed maximum limits
        if rightmax >np.shape(self.imagearray)[1]:
            rightmax = np.shape(self.imagearray)[1]
        ## car is 20 by 40 
        ## 200/5 = 40 pixels in vertical per lane

        diff_rows = np.shape(self.imagearray)[0]//self.lanes

        # total_visibility = 30

        for i in range(20):
            for j in range(rightmax-leftmax):

                self.imagearray[i+10 + lanenumber* diff_rows][leftmax+j] = value

    def shadewhere_truck(self, position, lanenumber, value):
        ## 4 units for the car
        leftmax = int(math.floor(position - 50))
        rightmax = int(math.floor(position + 50))
        if leftmax < 0:
            leftmax = 0
        ## cannot exceed maximum limits
        if rightmax >np.shape(self.imagearray)[1]:
            rightmax = np.shape(self.imagearray)[1]
        ## car is 20 by 40 
        ## 200/5 = 40 pixels in vertical per lane

        diff_rows = np.shape(self.imagearray)[0]//self.lanes

        # total_visibility = 30

        for i in range(20):
            for j in range(rightmax-leftmax):

                self.imagearray[i+10 + lanenumber* diff_rows][leftmax+j] = value




    def visualize_objects(original_func):
        def wrapper(self,*args,**kwargs):
            arr = original_func(self)
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    print(arr, end = " ")
                print()
            return arr
        return wrapper 

    # @visualize_objects
    def visible_game_objects(self):

        self.searchforPlayerCar() ## fill playerlanses and  playercarposition
        playercar = self.Game[self.playerlanes][self.playercarposition]
        visible_game_objects = []

        for i in range(len(self.Game)):
            ## 5 rows
            visible_game_objects.append([])
        for i in range(len(self.Game)):

            for j in range(len(self.Game[i])):


                if self.Game[i][j].GetPos() >= playercar.GetPos() - self.back:
                    ## inside visibility range

                    if self.Game[i][j].GetPos() <= playercar.GetPos():

                        visible_game_objects[i].append(self.Game[i][j])

                    if (self.Game[i][j].GetPos() == playercar.GetPos()\
                         and isinstance(self.Game[i][j],Vehicles.PlayerCar)) :

                        self.gameplayerindexvert = i ## index of player car in visibility game objects

                        self.gameplayerindexhorz = len(visible_game_objects[i]) - 1 ## most impt
                        visible_game_objects[i].append(self.Game[i][j])

                if self.Game[i][j].GetPos() <= playercar.GetPos() + self.front:
                    if self.Game[i][j].GetPos() > playercar.GetPos():
                        visible_game_objects[i].append(self.Game[i][j])


        return visible_game_objects       


    def image_from_visible_objects(self):
        self.imagearray = np.zeros(shape=(np.shape(self.imagearray)[0],np.shape(self.imagearray)[1])) ## display image array
        # self.imagearray.fill(0.01)
        # print("fun_call")

        vgo = self.visible_game_objects() #visible_game_objects 
        #player_car_index_in_visible_game_obj
        pcivgo = vgo[self.gameplayerindexvert][self.gameplayerindexhorz]
        
        for i in range(len(vgo)):
            for j in range(len(vgo[i])):

                if isinstance(vgo[i][j],Vehicles.Truck):
                        # print("truck",vgo[i][j].GetVel())
                        posn = (vgo[i][j].GetPos() - pcivgo.GetPos())* 10 + self.back * 10
                        # self.shadewhere_truck(posn,i,2.75)
                        self.shadewhere_truck(posn,i,vgo[i][j].GetVel())
                        continue                        


                if isinstance(vgo[i][j],Vehicles.PlayerCar):
                    posn =  self.back * 10
                    # print("player_car",self.gameplayerindexvert," ",vgo[i][j].GetVel()," ", vgo[i][j].GetPos())
                    self.shadewhere(posn,self.gameplayerindexvert,vgo[i][j].GetVel())
                    continue

                if isinstance (vgo[i][j], Vehicles.Car):
                    ## check if left or right
                        posn = (vgo[i][j].GetPos() - pcivgo.GetPos())* 10 + self.back * 10
                        # print("other_cars",vgo[i][j].GetVel())
                        # print(vgo[i][j].GetVel())
                        self.shadewhere(posn,i,vgo[i][j].GetVel())    


    def runGame(self, action, greedy):

        #if self.i == 0:
         #   temp = [0,0]
        #else:
        # temp = [0,0]
        ## temp is 0,-1 or 1
        temp = self.updateGameArray(action)
        # if greedy == True and temp[0] != 0:
        #     self.Game=self.temprestore
        #     return "REDO", 0, 0, 0

        ## convert 2d numpy image into 3 channel image
        # instead of returning self.imagearray  as observations return stacked_image 
        # stacked_img = np.stack((self.imagearray,)*3, axis=-1)

        # print("temp", temp)
        if temp[0] == 0:
            ##  imagearray , gamestatus : running: 0 , crashed: -1, win: 1, reward, done status: T OR F, ego_velocity
            return self.imagearray, temp[0], temp[1], False,self.Game[self.playerlanes][self.playercarposition].GetVel()
            # return  self.imagearray, temp[0], temp[1], False
            #self.i = 1
        else: 
            ## bot has crashed or bot has successfully cross 500 restart game
            #self.i=0
            # print("cancelled for now restart_simulation")
            tempcar = self.Game[self.playerlanes][self.playercarposition].GetVel()
            self.populateGameArray()
            # observation, reward, smallreward, done, velocity
            ## negative velocity is the reward

            return self.imagearray, temp[0], 0, True, temp[0] * tempcar
            # return self.imagearray,temp[0],0,True



def print_array(arr):
    count  = 0
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 2.75:
                print(arr[i][j], end = " ") 
                            

# def main():
#     # display = True
#     game = GameV1()
#     # game.overlay_image()
#     game.populateGameArray() ## 1 time

#     # for i in range(len(game .Game)):
#     #     for j in range(len(game.Game[i])):
#     #         print(game.Game[i][j]," ")
#     #     print(" ")


#     gameover = False
#     # cv2.namedWindow("game images")
#     # game.change_climate
#     # test = 0;
#     # game.give_img = True
#     # count  = 0
#     # accelerate = 1
#     count = 0 
#     while True:
#         action = int(input("please_input_action"))
#         if (action == "end"):
#             break
#         count += 1
#         print("count", count)

#         # action  = 0
#         # if (count % 40):
#         #     action = 1
#         game.give_img = False
#         a,b,c,d,e = game.runGame(action, False) ## for now no populate game array

#         # data = 255 * data # Now scale by 255
#         # img = data.astype(np.uint8)

#         # if ((b ==1) or (b == -1)):
#         #     break

#         img = a

#         # print_array(game.imagearray)
#         cv2.imshow("window",img)
#         # cv2.imshow("window",a)
#         key = cv2.waitKey(100)



        

#         # a,b,c,d,e = game.runGame(0,False)
        
#         print(type(b))
#         print(type(c))
#         print(a.dtype)
        # accelerate+=1
        # if accelerate%13 == 0:
            # print("accelerate")
            # a,b,c,d,e = game.runGame(1,False)




        # if (b==1):
        #     count+=1
        #     print("winning" + str(count))
        # if (b==-1):
        #     cv2.imshow(" ",game.imagearray)
        #     key = cv2.waitKey(3000)
        #     file_path = ''
        #     file_name = '\\test_collision\\'+ time.strftime('%Y_%m_%d_%H_%M_%S') + '.png'

        #     s = cv2.imwrite('F:\sim\Simulator_Presentation' + '\\test_collision\\'+ time.strftime('%Y_%m_%d_%H_%M_%S') + '.png', game.imagearray)
        #     print(s, os.path.join(file_path, file_name))
        #     game.populateGameArray()



            # print("ssss",s )


            # print("losing")


        # key = cv2.waitKey(3000)#pauses for 3 seconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        # cv2.destroyAllWindows()

        # print("visibility_positions")
        # game.print_array(game.createImageList())

  
        # game.playercarposition();
        # print("true position playercar",game.Game[game.playerlanes][game.playercarposition])

        # print(game.Game[game.playerlanes][game.playercarposition].GetPos())
        # game.print_positions_array()
        # game.give_me_all_positions() ## in player lane

        

    # while gameover == False:
    #    temp = game.updateGameArray(0)
    #    if temp != None:
    #        gameover = True
    #        print(temp)
    # cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


# g1 = GameV1(True)
# g1.overlay_image()
# g1.populateGameArray()
# g1.Get_Vehicle_Objects()
# g1.
# sum = 0;
# for i in range(0,5):
#     sum+=len(g1.Game[i])
# print(sum)

# for i in range(5):
#     g1.updateGameArray(0)
#     g1.updateGameArray(0)    
#     g1.updateGameArray(0)
#     g1.updateGameArray(0)
#     g1.updateGameArray(0)


# ## proof that local variables point to the same element in function
# arr = []
# for i in range(13):
#     arr.append(Vehicles.Car(0,2))

# update = arr[0]
# print(arr[0].GetPos())
# update.updatePos()
# update.updatePos()
# update.updatePos()
# update.updatePos()
# update.updatePos()
# print(arr[0].GetPos())

# subarray = []
# g1 = GameV1(True)
# g1.populateGameArray()
# g1.updateGameArray(0)
# subarray=g1.createImageList()

# print("subarray")
# for i in range(len(subarray)):
#     for j in range(len(subarray[i])):
#         print(subarray[i][j], end = " ")
#     print(" ")
# print(" ")
# print("len_subarray ", len(subarray))
# print("len_subarray[0] ", len(subarray[0]))


