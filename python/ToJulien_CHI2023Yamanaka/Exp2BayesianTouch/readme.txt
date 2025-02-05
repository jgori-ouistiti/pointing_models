participantNumber: from 0 to 17

bias: written as {Fast, Neutral, Accurate}

set: Set number (each set includes 27 trials)

trial: Trial number within a set

attempt: Tap attempt number within a trial (starting from 0). If the first tap attempt on a target results in an error, an additional row is added, whose attempt is 2. attempt could be then 3, 4, and so on.

touchX: Touch coordinate (X), in pixels

touchY: Touch coordinate (Y), in pixels

targetW: Target diameter, in mm

targetX: Target center coordinate (X), in pixels

targetY: Target center coordinate (Y), in pixels

distractorW: Diameter of the distractor, in mm

distractorP: Distractor placement

distractorX1 to X4 and Y1 to Y4: Center coordinates of distractors. When there are only two distractors, X3, Y3, X4, and Y4 are set to -1. Units: pixels

time: Time from successfully tapping the previous target to the next tap, in ms

error: Success or error judgment based on the Visual Boundary Criterion. 0 = success, 1 = error

isPrac: Indicates if it is a practice trial (the first two sets; set = 0, 1 are practice trials). 0 = main trial for data collection, 1 = practice trial

dist: Distance between the target center and the tap coordinate, in mm

Note: In the first trial of each set, the starting button was set at the center of the iPad screen (X = 1024, Y = 1366).
pixel-to-mm conversion: For this experiment, 264/25.4 [pixels/mm] was used.

