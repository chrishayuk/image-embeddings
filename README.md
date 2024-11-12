## installing qdrant
the simplest method of installing qdrant is to use the docker image.
to get the image run.

```bash
docker pull qdrant/qdrant
```

### run qdrant
once you've pulled the docker image, you can run it with the following command

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### check qdrant dashboard
to check the collection, you can go to the ui dashboard

http://localhost:6333/dashboard

## creating the initial index
to index a file on startup, you can create the initial index with the following command

```bash
python create_index.py
```

## adding a new image
to index a file on startup, you can create the initial index with the following command

```bash
python add_image.py --image_url "https://imgs.search.brave.com/T1Se6a_wzxbCKPNE0J8_PSy-QasoAdfJo4U7hjgYqEs/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/ZnJlZS1waG90by9i/ZWF1dGlmdWwtcGV0/LXBvcnRyYWl0LWRv/Z18yMy0yMTQ5MjE4/NDUwLmpwZz9zZW10/PWFpc19oeWJyaWQ"
```

## find a similar image
to find a similar image you can use the following command

```bash
python find_similar.py --image_url "https://imgs.search.brave.com/PHIRDEbjFhntV_Kbf_KdomVNifU36_T_Mkq7sMKBmWA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvNjI2/NDY0MTU4L3Bob3Rv/L2NhdC13aXRoLW9w/ZW4tbW91dGguanBn/P3M9NjEyeDYxMiZ3/PTAmaz0yMCZjPVFy/OURDVmt3S21fZHpm/amtlTjVmb0NCcDdj/M0VmQkZfaTJBMGV0/WWlKT0E9"
```

