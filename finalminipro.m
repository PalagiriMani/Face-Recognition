clc; clear; close all;

%% ===================== Student Registration with Image Capture =====================
dataset_path = 'student_images';
if ~exist(dataset_path, 'dir')
    mkdir(dataset_path);
end

% Define student records file

student_records_file = 'student_records.xlsx';

% Check if student records file exists, else create it
if isfile(student_records_file)
    student_data = readtable(student_records_file, 'TextType', 'string');
else
    student_data = table([], [], 'VariableNames', {'Name', 'ID'});
    writetable(student_data, student_records_file);
end

% Get student details
student_name = input('Enter Student Name: ', 's');
student_id = input('Enter Student ID: ', 's');

% Check if student already exists
if any(strcmp(student_data.ID, student_id))
    disp('Student already registered! Registration aborted.');
else
    % Create student folder
    folder_path = fullfile(dataset_path, [student_name '_' student_id]);
    mkdir(folder_path);

    % Capture images using webcam
    cam = webcam;
    num_images = 10;
    for i = 1:num_images
        img = snapshot(cam);
        
        % Convert to grayscale if necessary
        img_gray = im2gray(img);
        
        filename = fullfile(folder_path, sprintf('%s_%d.jpg', student_name, i));
        imwrite(img_gray, filename);
        imshow(img_gray);
        title(['Captured Image: ' num2str(i)]);
        pause(0.5);
    end
    clear cam;
    
    % Update student records
    new_student = table(string(student_name), string(student_id), 'VariableNames', {'Name', 'ID'});
    student_data = [student_data; new_student];
    writetable(student_data, student_records_file);
    
    disp('Student registered and images captured successfully!');
end

%% ================= Face Recognition & Attendance Marking =================
clc; close all;
faceDetector = vision.CascadeObjectDetector();
cam = webcam;

dataset_path = 'student_images';
imageFiles = dir(fullfile(dataset_path, '**', '*.jpg'));
knownFaces = {};
studentNames = {};

for i = 1:length(imageFiles)
    img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));

    % Convert to grayscale safely
    imgGray = im2gray(img);
    
    face = step(faceDetector, imgGray);
    
    if ~isempty(face)
        x = max(1, face(1,1));
        y = max(1, face(1,2));
        w = min(face(1,3), size(imgGray, 2) - x);
        h = min(face(1,4), size(imgGray, 1) - y);
        
        knownFaces{end+1} = imresize(imgGray(y:y+h, x:x+w), [100 100]);
        studentNames{end+1} = erase(imageFiles(i).folder, dataset_path);
    end
end

% Load or initialize attendance file
attendanceFile = 'attendance.xlsx';
if isfile(attendanceFile)
    attendance_data = readtable(attendanceFile, 'TextType', 'string');
    expectedVars = {'Name', 'ID'};
    if ~isequal(attendance_data.Properties.VariableNames, expectedVars)
        disp('Attendance file structure incorrect. Resetting format.');
        attendance_data = table([], [], 'VariableNames', expectedVars);
    end
else
    attendance_data = table([], [], 'VariableNames', {'Name', 'ID'});
    writetable(attendance_data, attendanceFile);
end

seen_students = {};

while true
    frame = snapshot(cam);
    
    % Convert to grayscale safely
    grayFrame = im2gray(frame);
    
    bbox = step(faceDetector, grayFrame);
    
    if ~isempty(bbox)
        for i = 1:size(bbox,1)
            x = max(1, bbox(i,1));
            y = max(1, bbox(i,2));
            w = min(bbox(i,3), size(grayFrame, 2) - x);
            h = min(bbox(i,4), size(grayFrame, 1) - y);

            detectedFace = imresize(grayFrame(y:y+h, x:x+w), [100 100]);
            
            bestMatch = 'Unknown';
            minError = inf;
            studentID = '';

            for j = 1:length(knownFaces)
                error = immse(detectedFace, knownFaces{j});
                if error < minError
                    minError = error;
                    bestMatch = studentNames{j};
                end
            end

            if ~strcmp(bestMatch, 'Unknown')
                splitData = strsplit(bestMatch, '_');
                student_name = splitData{1};
                studentID = splitData{2};

                if ~ismember(studentID, seen_students)
                    newEntry = {student_name, studentID};
                    newEntryTable = cell2table(newEntry, 'VariableNames', {'Name', 'ID'});

                    % Ensure table structure matches before concatenation
                    if isempty(attendance_data)
                        attendance_data = newEntryTable;
                    else
                        attendance_data = [attendance_data; newEntryTable]; %#ok<AGROW>
                    end

                    seen_students{end+1} = studentID;
                end
            end
            
            frame = insertObjectAnnotation(frame, 'rectangle', bbox(i,:), student_name, 'Color', 'green');
        end
    end

    imshow(frame);
    title('Face Recognition Attendance System');
    
    if waitforbuttonpress
        break;
    end
end

clear cam;
disp('Attendance Recorded Successfully!');

% Remove duplicates and save attendance
[~, uniqueIdx] = unique(attendance_data(:,1:2), 'rows', 'stable');
attendance_data = attendance_data(uniqueIdx, :);

writetable(attendance_data, attendanceFile);
disp('Attendance updated successfully!');
