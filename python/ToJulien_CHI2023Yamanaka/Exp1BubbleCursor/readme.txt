Note: Outliers have already been removed. Specifically, as stated in our paper, "We removed 53 outlier trials (0.18%) where the movement distance for the first click point was shorter than A/2 or the distance to the target center was longer than 2W for Point and 2EW for Bubble." Initially, the dataset included a total of 29,808 trials, but 53 trials were excluded based on these criteria, so this csv file contains data for 29,755 trials.

participantNumber: from 0 to 11

bias: 0 = Fast, 1 = Neutral, 2 = Accurate

isBubble: FALSE = Point Cursor, TRUE = Bubble Cursor

set: Indicates the set number for each bias Å~ cursor condition. Each participant completed a total of 18 sets (2A Å~ 3W Å~ 3EW/W = 18 conditions). This just preserves the experimental sequence and is likely unnecessary information for you.

order: Indicates the target number within a set (excluding the starting target). For example, order = 1 corresponds to data where the target "1" (which acts as the starting button, as shown in Figure 2b of the paper) is clicked, followed by a click on the target "2". Thus, for order = 1, with coordinates based on the origin at the screen center (x = 0, y = 0), the first-trial target (i.e., target "2") is located much left (x = -198) and slightly above it (y = 27). Since the experiment was implemented in Unity, note that the y-axis points upwards.

x, y: x and y coordinates of the first click in that trial

success: 1 = success, 0 = error

time: Time taken for the first click in that trial, measured in seconds.