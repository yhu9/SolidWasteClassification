
import cv2
import numpy as np
import os
import sys

names= ['treematter','plywood','cardboard','black bags','trash bags', 'plastic bottles']
treematter_mask = [0,0,255]
plywood_mask = [0,255,0]
cardboard_mask = [255,0,0]
blackbag_mask = [255,255,0]
trashbag_mask = [255,0,255]
bottles_mask = [0,255,255]
mask_colors = [treematter_mask,plywood_mask,cardboard_mask,blackbag_mask,trashbag_mask,bottles_mask]
BORDER = 28

def dice(img,gt,fout='dice_output.txt',writemode='w'):
    accs = []
    dices = []
    ratios = []
    with open(fout,writemode) as fo:
        for cat,mask in zip(names,mask_colors):
            TP = float(np.count_nonzero(np.logical_and(np.all(img == mask,axis=2), np.all(gt == mask,axis=2))))
            TN = float(np.count_nonzero(np.logical_and(np.logical_not(np.all(img == mask,axis=2)), np.logical_not(np.all(gt == mask,axis=2)))))
            FP = float(np.count_nonzero(np.logical_and(np.all(img == mask,axis=2),np.logical_not(np.all(gt == mask,axis=2)))))
            FN = float(np.count_nonzero(np.logical_and(np.logical_not(np.all(img == mask,axis=2)),np.all(gt == mask,axis=2))))
            P = float(TP + FN)
            N = float(TN + FP)


            #for debugging purposes
            #binary1 = np.array(np.all(img == mask,axis = 2),dtype=np.uint8)
            #binary2 = np.array(np.all(gt == mask,axis = 2),dtype=np.uint8)
            #b1_and_b2 = np.array(np.logical_and(binary1,binary2),dtype=np.uint8)
            #binary1[binary1 == 1] = 255
            #binary2[binary2 == 1] = 255
            #b1_and_b2[b1_and_b2 == 1] = 255
            #cv2.imshow('binary1',cv2.resize(binary1,(500,500),interpolation=cv2.INTER_CUBIC))
            #cv2.imshow('binary2',cv2.resize(binary2,(500,500),interpolation=cv2.INTER_CUBIC))
            #cv2.imshow('b1_and_b2',cv2.resize(b1_and_b2,(500,500),interpolation=cv2.INTER_CUBIC))
            #cv2.waitKey(0)

            if (P+N) == 0:
                print('error with p + n')
            elif(P) == 0:
                print('error with p')
                P += 1
            elif(N) == 0:
                print('error with n')
                N += 1

            if (TP + FP) == 0:
                FP += 1

            PREC = (TP) / (TP + FP)
            ACC = (TP + TN) / (P + N)
            SENS= TP / P
            SPEC= TN / N

            DICE = TP / (P + FP)

            ratios.append(100 * np.count_nonzero(np.all(gt == mask,axis=2)) / (P + N))
            accs.append(ACC)
            dices.append(DICE)

            print('-----category: %s ------' %cat)
            print('True Positive: %f' % TP)
            print('True Negative: %f' % TN)
            print('False Positive: %f' % FP)
            print('False Negative: %f' % FN)
            print('Positive: %f' % P)
            print('Negative: %f\n' % N)
            print('PRECICSION: %f' % PREC)
            print('ACCURACY: %f' % ACC)
            print('SENSITIVITY: %f' % SENS)
            print('SPECIFICITY: %f' % SPEC)
            print('ACCURACY: %f' % ACC)
            print('DICE: %f' % DICE)
            print('--------------')

            fo.write('--------' + cat + '--------\n\n\n')
            fo.write('True Positive: %f\n' % TP)
            fo.write('True Negative: %f\n' % TN)
            fo.write('False Positive: %f\n' % FP)
            fo.write('False Negative: %f\n' % FN)
            fo.write('Positive: %f\n' % P)
            fo.write('Negative: %f\n\n' % N)
            fo.write('PRECICSION: %f\n' % PREC)
            fo.write('SENSITIVITY: %f\n' % SENS)
            fo.write('SPECIFICITY: %f\n' % SPEC)
            fo.write('ACCURACY: %f\n' % ACC)
            fo.write('DICE: %f\n\n\n' % DICE)

        total_acc = 0
        total_dice = 0
        for r,acc,dice in zip(ratios,accs,dices):
            total_acc += r * acc
            total_dice += r * dice

        print('TOTAL ACCURACY: %f' % total_acc)
        print('TOTAL DICE SCORE: %f' % total_dice)
        fo.write('TOTAL ACCURACY: %f\n' % total_acc)
        fo.write('TOTAL DICE SCORE: %f\n' % total_dice)
        fo.write('---------------------------------------------------\n')

    return total_acc,total_dice,dices


#main function
if __name__ == '__main__':
    #maker sure of correct sys args

    #WHEN DICING A directory
    if len(sys.argv) == 4 and sys.argv[1] == 'all':
        dir1 = sys.argv[2]
        dir2 = sys.argv[3]
        if not os.path.isdir(dir1):
            print('%s is not a directory!' % dir1)
            sys.exit()
        if not os.path.isfile(dir2):
            print('$s is not a file!' % dir2)
            sys.exit()
        #output to results directory
        if not os.path.exists('results'):
            os.makedirs('results')

        gt = cv2.imread(dir2,cv2.IMREAD_COLOR)
        dice_max = 0
        acc_max = 0
        best = ''
        cat_max = [0,0,0,0,0,0]
        cat_best = ['','','','','','']
        for fname in os.listdir(dir1):
            fin = os.path.join(dir1,fname)
            fname = "RESULTS_" + str(os.path.splitext(os.path.basename(fname))[0]) + ".txt"
            fileout = os.path.join('results',fname)

            #read the image
            img = cv2.imread(fin,cv2.IMREAD_COLOR)

            #dice the image according to ground truth
            print("READING: %s" % fileout)
            acc_score,dice_score,cat_dices = dice(img,gt,fout=fileout)

            #save the file name for the best category file, and the best overall file
            if dice_score > dice_max:
                dice_max = dice_score
                acc_max = acc_score
                best = fin

            for i,cat_score in enumerate(cat_dices):
                if cat_score > cat_max[i]:
                    cat_max[i] = cat_score
                    cat_best[i] = fin

        #write to overall output file. INDIVIDUAL RESULTS are written in the dice function to fout
        outputfile = 'results/' + os.path.basename(dir1) + '_overall.txt'
        with open(outputfile,'w') as fo:
            for val,fname,cat_name in zip(cat_max,cat_best,names):
                fo.write('%s : %s ----> %f \n' % (cat_name, fname, val))

                print('%s : %s ----> %f \n' % (cat_name, fname, val))
            fo.write('%s : %s ----> %f \n' % ('BEST OVERALL DICE', best, dice_max))
            fo.write('%s : %s ----> %f \n' % ('ACCOMPANYING ACC', best, acc_max))
            print('%s : %s ----> %f \n' % ('BEST OVERALL', best, dice_max))

    #WHEN DICING A SINGLE IMAGE
    elif len(sys.argv) == 3:
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]

        #check if directory exists then read the images
        if os.path.exists(dir1) and os.path.exists(dir2):
            img = cv2.imread(dir1,cv2.IMREAD_COLOR)
            gt = cv2.imread(dir2,cv2.IMREAD_COLOR)
            h,w,d = img.shape
            h2,w2,d2 = gt.shape
            if h != h2 or w != w2:
                gt = cv2.resize(gt,(h,w),interpolation = cv2.INTER_CUBIC)

            #output to results directory
            if not os.path.exists('results'):
                os.makedirs('results')

            #create file name
            fname = "RESULTS_" + str(os.path.splitext(os.path.basename(sys.argv[1]))[0]) + ".txt"
            fout = os.path.join('results',fname)

            #apply dice score calculation
            h_low = BORDER
            h_high = h - BORDER - 1
            w_low = BORDER
            w_high = w - BORDER - 1
            acc_score,dice_score, cat_dice = dice(img[h_low:h_high,w_low:w_high,:] ,gt[h_low:h_high,w_low:w_high,:],fout)

        else:
            print("PATH DOES NOT EXIST: \n\t%s, \n\t%s" %(sys.argv[1],sys,argv[2]))
            sys.exit()
    else:
        print("wrong number of arguments")
        print("expecting 2")

