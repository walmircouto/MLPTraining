import org.apache.spark._
import spark.implicits._
import java.io._

val metamodel = spark.read.json("examples/doutorado/metamodel/javaMetamodel.json")

val metamodelItemsAmount = metamodel.select(distinct(*)).count

val nameItemMetamodel = "empty"

for(int n = 0; n < metamodelItemsAmount; n++) {

	nameItemMetamodel = nameItemMetamodel + metamodel.select("element.name").get(n)	

}

val binaryDigitsAmount = exp(2ˆn = metamodelItemsAmount).toInt

val MLPVectorX = "empty"

val elementNameBinary = "empty"


for (int n = 0; n < nameItemMetamodel; n++) {

	MLPVectorX(n) = create.binary(nameItemMetamodel.element.name, binaryDigitsAmount)

	elementNameBinary = nameItemMetamodel.element.name + MLPVectorX(n)
}

MLPVectorX(n).rdd.saveFile("examples/doutorado/metamodel/MLPVectorX")

elementNameBinary.rdd.saveFile("examples/doutorado/metamodel/elementNameBinary")

System.exit(0)