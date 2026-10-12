CREATE DATABASE IF NOT EXISTS university_db;

USE university_db;

CREATE TABLE departments (
    department_id INT AUTO_INCREMENT PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    department_code VARCHAR(10) NOT NULL UNIQUE
);

CREATE TABLE students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    gender ENUM('Male', 'Female', 'Other') NOT NULL,
    date_of_birth DATE NOT NULL,
    admission_year INT NOT NULL,
    current_semester INT NOT NULL,
    cgpa DECIMAL(4, 2) NOT NULL,
    department_id INT NOT NULL,

    CONSTRAINT fk_student_department
        FOREIGN KEY (department_id)
        REFERENCES departments(department_id)
);

CREATE TABLE courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_code VARCHAR(20) NOT NULL UNIQUE,
    course_name VARCHAR(150) NOT NULL,
    credits INT NOT NULL,
    semester INT NOT NULL,
    department_id INT NOT NULL,

    CONSTRAINT fk_course_department
        FOREIGN KEY (department_id)
        REFERENCES departments(department_id)
);

CREATE TABLE enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    academic_year VARCHAR(10) NOT NULL,
    semester_number INT NOT NULL,
    marks DECIMAL(5, 2),
    grade VARCHAR(2),
    attendance_percentage DECIMAL(5, 2),

    CONSTRAINT fk_enrollment_student
        FOREIGN KEY (student_id)
        REFERENCES students(student_id),

    CONSTRAINT fk_enrollment_course
        FOREIGN KEY (course_id)
        REFERENCES courses(course_id),

    CONSTRAINT unique_enrollment
        UNIQUE (
            student_id,
            course_id,
            academic_year,
            semester_number
        )
);