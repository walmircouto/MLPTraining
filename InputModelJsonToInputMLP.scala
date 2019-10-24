import org.apache.spark._
import spark.implicits._
import java.io._

val inputModel = spark.read.json("examples/doutorado/metamodel/InputModel.json")

val ReferenceElementBinary = spark.read.filetext("examples/doutorado/metamodel/elementNameBinary")

val inputModelItemsAmount = inputModel.select(distinct(*)).count

val MLPTextFile = "empty"

val elementName = "empty" 

for(int n = 0; n < (metamodelItemsAmount - 1); n++) {

	elementName = ReferenceElementBinary.select("name", "binary").filter($"name" === inputModel.name.head().get(n))

	if (elementName(n) =!= "null") { 

			MLPTextFile = MLPTextFile + elementName(n).binary 			
		
	}
}

MLPTextFile.rdd.saveFile("examples/doutorado/metamodel/MLPTextFile")

System.exit(0)