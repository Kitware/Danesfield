# -*- coding: utf-8 -*-

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

"""
Created on 25 May 2018
@author: kellie.corona
"""


def gen_kw18(polygons, poly_types=None, fname_base="citygml_kw18"):
    """
    This function takes polygons as a dictionary mapping ID to a list of (x,y)
    tuples and returns the kw18 files necessary for viewing in WAMI-Viewer.

    Parameters
    ----------
    polygons : dictionary
        dictionary containing polygon verticies as a list of tuples mapped to ID
        {0: [(x01, y01),...,(x0i,y0i)], ..., N: [(xN0,yN0),...,(xNj,yNj)]}
    poly_types: dictionary
        optional input, dictionary containing names for polygons
        {0: 'Building', ... , N: 'Road'}

    Returns
    -------
    Writes out kw18, regions and types files corresponding to inputs.
    """
    if poly_types:
        with open(fname_base+".kw18.types", "w") as ftypes:
            for p_id, p_val in poly_types.items():
                ftypes.write(str(p_id)+" "+str(p_val)+"\n")

    with open(fname_base+".kw18.regions", "w") as fregions, \
            open(fname_base+".kw18", "w") as fkw18:
        # kw18.regions file structure is one line per frame per track where each column is:
        # 1:track-ID 2:0 3:1 4:number-of-verticies 5++:x-coordinate y-coordinate
        # kw18 file structure is one line per frame per track where each column is:
        fkw18.write("#1:Track-id 2:Track-length 3:Frame-number 4:Tracking-plane-loc(x)"
                    " 5:Tracking-plane-loc(y) 6:velocity(x) 7:velocity(y) 8:Image-loc(x)"
                    " 9:Image-loc(y) 10:Img-bbox(TL_x) 11:Img-bbox(TL_y) 12:Img-bbox(BR_x)"
                    " 13:Img-bbox(BR_y) 14:Area 15:World-loc(longitude) 16:World-loc(latitude)"
                    " 17:World-loc(altitude) 18:timesetamp 19:track-confidence \n")
        for p_id, polygon in polygons.items():
            x_vert, y_vert = [], []
            # constructing regions file line
            regions_str = str(p_id) + " 0 1 " + str(len(polygon))
            for vert in polygon:
                regions_str = regions_str + " " + str(vert[0]) + " " + str(vert[1])
                x_vert.append(vert[0])
                y_vert.append(vert[1])
            fregions.write(regions_str+"\n")
            # constructing kw18 file line
            img_xloc = min(x_vert)+(max(x_vert)-min(x_vert))/2
            img_yloc = min(y_vert)+(max(y_vert)-min(y_vert))/2
            kw18_str = " ".join([str(p_id), "1 0 0 0 0 0", str(img_xloc),
                                str(img_yloc), str(min(x_vert)),
                                str(min(y_vert)), str(max(x_vert)),
                                str(max(y_vert)), "-1 444 444 0 0 -1\n"])
            fkw18.write(kw18_str)


if __name__ == '__main__':
    # citygml_poly = {0: [(30,10),(40,40),(20,40),(10,20),(30,10)],
    #                 1: [(35,10),(45,45),(15,40),(10,20),(35,10)],
    #                 2: [(20,30),(35,35),(0,20),(20,30)]}
    # citygml_types = {1: "Building", 2: "Outter_Ring", 3: "Inner_Ring"}
    citygml_poly = {0: [(446, 293), (381, 273), (319, 281), (320, 327), (324, 367), (436, 363)],
                    1: [(1173, 207), (1098, 192), (1069, 218),
                        (975, 226), (1020, 487), (1232, 440), (1201, 189)],
                    2: [(1073, 261), (1077, 313), (1085, 315), (1080, 259)]}
    citygml_types = {0: "Office", 1: "Field", 2: "Row"}
    gen_kw18(citygml_poly, citygml_types)
