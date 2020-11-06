import os

import pandas as pd
import skvideo.io
import torch
import torchvision
import numpy as np


class DataSet:
    def __init__(self, file_name, video_path):
        self.file_name  = file_name  
        self.detections = pd.read_csv(file_name, header=None)
        self.query   = None
        self.gallery = None
        self.video   = skvideo.io.vread(video_path)
        self.reid    = None

    def get_suspicious_boxes(self, group, disappear=True, all_suspicious=False, 
                         xs=(10, 950), ys=(10, 530)):
        '''
        Accepts a pandas groupby object (grouped by a particular person
        id) and returns suspicious boxes produced by the model. 

        Parameters:
            group (pandas groupby object): Grouped by person ID, must have attributes
            frame, topx, offsetx, topy, and offsety where...
                - frame is the 1-based indexing frame number
                - topx and topy are the x and y coordinates representing a corner of 
                a bounding box
                - offsetx and offsety are the bounding box width and height respectively
            disappear (Boolean): If True, finds all boxes that disappear permanently but
            their last occurrence is not on the border of the video. If disappear is 
            False, then a suspicious region is a box that appears for the first time
            in the middle of the video frame (not on image edge)
            all_suspicious (boolean): If True, then all of the suspicious samples are 
            returned (aka, the entire group). Else, just the last image of a 
            suspicious disappearing person or the first image of a suspicious
            appearing person is returned
            xs: Tuple of the smallest and largest possible x coordinates for us to not
                to consider a bounding box to be along the border of an image
            ys: Tuple of the smallest and largest possible y coordinates for us to not 
                to consider a bounding box to be along the border of an image
        '''
        if disappear:
            suspicious_image = group.iloc[-1]
            idx = 1500 # Index of last image in video

        elif not disappear: # Looking for appearing image
            suspicious_image = group.iloc[0]
            idx = 0  # Index of first image in video
        
        else:
            raise Exception('Disappear parameter must be a boolean')

        # Image border perimeter
        x1, x2 = xs
        y1, y2 = ys
        
        # This person on border or at beginning/end of video
        if (suspicious_image['topx'] < x1 or \
            (suspicious_image['topx'] + suspicious_image['offsetx']) > x2 or \
            suspicious_image['topy'] < y1 or \
            (suspicious_image['topy'] + suspicious_image['offsety']) > y2) or \
            suspicious_image['frame'] == idx:  
            return -1

        if all_suspicious:
            return [[int(attribute) for attribute in list(image)] 
                                    for _, image in group.iterrows()]
        else:
            return [[int(attribute) for attribute in list(suspicious_image)]]
    
    def get_suspicious_group(self, disappear, all_suspicious, xs=(10, 950), ys=(10, 530)):
        '''
        Returns a list of the form [(p_id, [[<frame>, <topx>, <topy>, <width>, <height>], ...])]
        for either all boxes that match a suspicious id (if all_suspicious is true) or just one
        box (first box if disappear is False, last box if disappear is True).

        Parameters:
            disappear (Boolean): Gets suspicious disappearing box if set to true, else it gets a suspicious appearing
                box
            all_suspicious (Boolean): Whether we want all of the suspicious boxes that match a particular ID or just 
            one of them
            xs: Tuple of the smallest and largest possible x coordinates for us to not
                to consider a bounding box to be along the border of an image
            ys: Tuple of the smallest and largest possible y coordinates for us to not 
                to consider a bounding box to be along the border of an image
        '''
        suspicious = self.detections.groupby('id')[['frame', 'topx', 'topy', 
                                        'offsetx', 'offsety']].apply(
                                        get_suspicious_boxes, disappear=disappear,
                                        all_suspicious=all_suspicious, xs=xs, ys=ys)

        suspicious_group = [(pid, query[pid]) for pid in query.keys() 
                                                if query[pid] != -1]
        if disappear:
            self.query = suspicious_group
        else:       
            self.gallery = suspicious_group
    
    def display_suspicious_sample_images(self, sus_type='disappear', title='Sample of suspicious candidates'):
        '''
        Displays a sample of four suspicious individuals 

        Parameters:
            sus_type (String): One of 'disappear' or 'reappear'. If 'disappear', a sample of the query images is shown, 
                else, a sample of the gallery images is shown
            title (String): Title of the figure 
        '''
        
        if sus_type == 'disappear':
            if self.query is None:
                print('Ensure that you run \'get_suspicious_group\' for disappearing images to get query group')
                return
            suspicious = self.query
        
        elif sus_type == 'appear':
            if self.gallery is None:
                print('Ensure that you run \'get_suspicious_group\' for aappearing images to get gallery group')
                return
            suspicious = self.gallery
        
        else:
            print('\'sus_type\' must be one of \'disappear\' or \'appear\'')
            return
        
        # A few examples of people that disappear
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(title)
        random_sample = random.sample(suspicious, 4)

        for i in range(2):
            for j in range(2):
                id, ims = random_sample[2*i+j]
                frame, x, y, offsetx, offsety = ims[-1]

                frame_array = self.video[frame - 1]
                border = np.zeros(frame_array.shape)
                cv2.rectangle(border, (x, y), (x + offsetx, y + offsety), (255, 0, 0), 10)
                overlay = np.where(border != 0, border, frame_array).astype(np.uint8)

                ax[i][j].imshow(overlay)
                ax[i][j].set_title('pid: {}, frame: {}, p1: ({}, {}), p2: ({}, {})'
                                    .format(id, frame, x, y, x + offsetx, y + offsety))
        plt.show()
        
    def create_suspicious_video(self, num_additional=20, 
                             sus_type='disappear'):
        '''
        Creates a video of the suspicious sample.

        Parameters:
            num_additional (int): How many additional frames we want for the suspicious video
            sus_type (String): If 'disappear', then we want to display 'num_additional' frames after the last 
                time the person was re-identified, and if 'reappear', then we want to display 'num_additional' 
                frames before the first occurence that the person was identified
        '''

        if sus_type == 'disappear':
            if self.query is None:
                print('Ensure that you run \'get_suspicious_group\' for disappearing images to get query group')
                return
            suspicious = self.query
        
        elif sus_type == 'appear' and self.gallery is not None:
            if self.gallery is None:
                print('Ensure that you run \'get_suspicious_group\' for aappearing images to get gallery group')
                return
            suspicious = self.gallery
        
        elif sus_type not in ('disappear', 'appear'):
            print('\'sus_type\' must be one of \'disappear\' or \'appear\'')

        id, data = suspicious[random.randint(0, len(suspicious) - 1)]
        fig, ax = plt.subplots(figsize=(5, 8))
        fig.suptitle('Suspicious id: {}'.format(id))

        if sus_type == 'disappear':
            data = data[-30:]  # Grabs the last two seconds 

            video_gen = ((self.video[data[i][0] - 1], *data[i][1:]) 
                        for i in range(len(data)))

            last_frame = data[-1][0]

            # 'num_additional' frames after the last frame
            additional_frames = ((self.video[last_frame + i], None)
                                for i in range(num_additional) 
                                    if last_frame + i < len(video))
            video_gen = itertools.chain(video_gen, additional_frames)
            
        elif sus_type == 'appear':
            data = data[:30]  # Grabs the first two seconds

            video_gen = ((self.video[data[i][0] - 1], *data[i][1:]) 
                        for i in range(len(data)))

            first_frame = data[0][0]

            # 'num_additional' frames before the first frame
            additional_frames = ((self.video[first_frame + i], None)
                                for i in range(-num_additional, 0)
                                if last_frame + i >= 0)
            video_gen = itertools.chain(additional_frames, video_gen)

        else:
            raise Exception('sus_type must be one of \'disappear\' or \'appear\'')

        def update(i):
            print('On iteration: {}/{}'.format(i, len(data) + num_additional))
            next_item = next(video_gen)

            if next_item[1] == None:
                ax.imshow(next_item[0])
            
            else:
                video_frame, x, y, offsetx, offsety = next_item 
                bbox = np.zeros(video_frame.shape)
                cv2.rectangle(bbox, (x, y), (x + offsetx, y + offsety), (255, 0, 0), 10)
                overlay = np.where(bbox != 0, bbox, video_frame).astype(np.uint8)
                ax.imshow(overlay)

        ani = animation.FuncAnimation(fig, update, len(data) + num_additional - 1, 
                                        interval=50, blit=False, repeat=True)
        ani.save('person.gif', writer="imagemagick")
        plt.close()
                

def make_dataset(self):
    '''
    Creates a dataset with crops around the bounding box areas for the query and gallery
    '''

    if self.query is None or self.gallery is None:
        print('Cannot have a null query or gallery')
        return

    image_sets = [('query', self.query), 
                ('gallery', self.gallery)]

    for (image_type, image_set) in image_sets:
        if image_type == 'query':
            parent_folder = 'FairMOT/prbp/query'
        elif image_type == 'gallery':
            parent_folder = 'FairMOT/prbp/gallery'
        for pid, bboxes in image_set:
            pid_dir = '{}/{}'.format(parent_folder, pid)
            os.makedirs(pid_dir, exist_ok=True)
            for (frame, x, y, offsetx, offsety) in bboxes:
                frame_array = self.video[frame - 1]
                start_x = max(0, x)
                start_y = max(0, y)

                crop = frame_array[start_y:y + offsety, 
                                    start_x:x + offsetx]
                filename = '{}/{}_{}_{}_{}_{}.jpg'.format(pid_dir, frame, start_x, 
                                                            start_y, offsetx, offsety)
                cv2.imwrite(filename, crop)
    
    def sort_img(qf, ql, qc, gf, gl, gc):
        '''
        Given the query and gallery features, labels, and cameras, returns a sorted 
        index of best possible matches for a particular query, as well as the 
        probabilities of the matches

        qf, gf: Matrices of query and gallery features, respectively
        ql, gl: Vectors of query and gallery labels, respectively
        qc, gl: Vector of the camera id for each one of the query and gallery labels,
                respectively
        '''

        query = qf.view(-1,1)
        score = torch.mm(gf,query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        
        # Predict index
        index = np.argsort(score) 
        index = index[::-1]
        
        # Good index
        query_index = np.argwhere(gl==ql)
        
        # Same camera
        camera_index = np.argwhere(gc==qc)

        junk_index1 = np.argwhere(gl==-1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1) 

        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]
        return index, score
    
    def reidentify_all_query(result_mat_path, thresh=0):
        '''
        Given the path to the result matrix, performs subject re-idenitification

        Parameters:
            result_mat_path (String): Path to the path of results after performing subject re-identification
            thresh (float): Real number between 0 and 1, the min probability of re-identification
        '''

        result = scipy.io.loadmat(result_mat_path)
        query_features = torch.FloatTensor(result['query_f'])
        query_cams = result['query_cam']
        query_labels = result['query_label'][0]
        gallery_features = torch.FloatTensor(result['gallery_f'])
        gallery_cams = result['gallery_cam'][0]
        gallery_labels = result['gallery_label'][0]

        query_features = query_features.cuda()
        gallery_features = gallery_features.cuda()

        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join('prbp', x)) 
                                for x in ['gallery','query']}

        reid = dict()

        for q_id in range(len(query_labels)):
            query_label = query_labels[q_id]
            query_path, _ = image_datasets['query'].imgs[q_id]

        # Path in the form prbp/query/id/<frame>_<x>_<y>_<offsetx>_<offsety>.jpg
        # Looking to extract the frame value
        q_frame = query_path.split('/')[-1].split('_')[0]

        index, scores = sort_img(query_features[q_id], query_labels[q_id], 
                                    query_cams[q_id], gallery_features, 
                                    gallery_labels, gallery_cams)

        found_match = False
        g_id = 0

        # The lowest probability of match we allow, default is 0, but typically
        # you would want this to be higher (ex. 0.70)
        best_score = thresh   

        while g_id < 3 and not found_match:  # Limit the number of top matches to 3
            gallery_path, _ = image_datasets['gallery'].imgs[index[g_id]]

            # We want to ensure that the gallery image comes in a later frame than
            # the query image
            g_frame = gallery_path.split('/')[-1].split('_')[0]
            if int(g_frame) > int(q_frame):
                g_label = gallery_labels[index[g_id]]
                score = scores[index[g_id]]

                # We need to determine that the score of this gallery match is better
                # than the current best score
                if g_label not in reid or score > best_score:
                    reid[g_label] = (query_label, score)
                    found_match = True
                    best_score = score
            g_id += 1
        self.reid = reid
    
    def save_reid(self, path):
        '''
        Saves the updated re-identification data to the file specified by path

        Parameters:
            path (String): Location to save the newly re-identified data
        '''
        
        if self.reid is None:
            print('You must first perform re-identification')

        else:
            new_detections = pd.read_csv(self.file_name, header=None)
            new_detections.columns = ['frame', 'id', 'topx', 'topy', 'offsetx', 'offsety',
                                    0, 1, 2, 3]

            for g_id in self.reid:
                gallery = new_detections.loc[new_detections['id'] == g_id]['id'].index
                new_detections.loc[gallery, 'id'] = self.reid[g_id][0]

            new_detections.to_csv(path, header=False, index=False)
  