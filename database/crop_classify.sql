/*
Navicat MySQL Data Transfer

Source Server         : mh
Source Server Version : 50740
Source Host           : localhost:3306
Source Database       : crop_classify

Target Server Type    : MYSQL
Target Server Version : 50740
File Encoding         : 65001

Date: 2025-03-25 08:56:27
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for crops
-- ----------------------------
DROP TABLE IF EXISTS `crops`;
CREATE TABLE `crops` (
  `Cno` int(11) NOT NULL,
  `Cclass` varchar(255) NOT NULL,
  `Cdisaster` varchar(255) NOT NULL,
  `Cdescription` text NOT NULL,
  `Csolution` text NOT NULL,
  `Cpicture` longblob,
  `Csymptoms` text NOT NULL,
  PRIMARY KEY (`Cno`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for records
-- ----------------------------
DROP TABLE IF EXISTS `records`;
CREATE TABLE `records` (
  `Rno` int(11) NOT NULL AUTO_INCREMENT,
  `R_Uid` varchar(255) NOT NULL,
  `Rtime` datetime NOT NULL,
  `Rclass` varchar(255) NOT NULL,
  `Rdisaster` varchar(255) NOT NULL,
  `Rpicture` longblob NOT NULL,
  PRIMARY KEY (`Rno`),
  KEY `Record_Uid` (`R_Uid`),
  CONSTRAINT `Record_Uid` FOREIGN KEY (`R_Uid`) REFERENCES `users` (`Uid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `Uid` varchar(255) NOT NULL,
  `Uname` varchar(255) NOT NULL,
  `Upassword` varchar(255) NOT NULL,
  `Uheadshot` longblob NOT NULL,
  `Ugender` varchar(50) NOT NULL,
  `Uphone` varchar(255) NOT NULL,
  PRIMARY KEY (`Uid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
