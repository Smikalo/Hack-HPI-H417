#Importing the Nominatim geocoder class
from geopy.geocoders import Nominatim
import streamlit as st


def get_user_input(StLit=True):

    #making an instance of Nominatim class
    geolocator = Nominatim(user_agent="my_request")

    #address inputs
    if StLit:
        st.header("Enter addresses in Berlin")
        adr1 = st.text_input('Initial',
                            'Eislebener Str. 4') #, 10789 Berlin
        adr2 = st.text_input('Destination',
                            'S+U Friedrichstrasse') #10117 
                            #'Hauptbahnhof Europaplatz 1') #10178
        go = st.button('Calculate route')
    else:
        adr1 = 'Eislebener Str. 4' #, 10789 Berlin
        adr2 = 'Panoramastrasse 1A' #, , 10178 Berlin
        print(f' Using default coordinates\n {adr1}\n {adr2}')
        go = True
    if adr1.split(' ')[-1]!='Berlin':
        adr1 = adr1 + ' Berlin'
    if adr2.split(' ')[-1]!='Berlin':
        adr2 = adr2 + ' Berlin'

    if go:
        C1 = geolocator.geocode(adr1)
        C2 = geolocator.geocode(adr2)
        success=True
        if StLit:
            if C1 == None:
                st.write(f'Coordinates could not be found for {adr1}')
                success = False
            else:
                st.write(f' Coordinates found for {adr1}: {C1.longitude}, {C1.latitude}')
            if C2 == None:
                st.write(f'Coordinates could not be found for {adr2}')
                success = False
            else:
                st.write(f' Coordinates found for {adr2}: {C2.longitude}, {C2.latitude}')
        else:
            print(f' Coordinates found for {adr1}: {C1.longitude}, {C1.latitude}')
            print(f' Coordinates found for {adr2}: {C2.longitude}, {C2.latitude}')
        return C1, C2, success
    else:
        return 0, 0, False